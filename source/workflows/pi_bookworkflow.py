"""
Book summary workflow built on the `pi` coding agent (https://pi.dev) instead of
LangGraph/LangChain + a vector store. Re-implements an earlier LangGraph +
vector-store book-summary pipeline without either dependency:

  - No LangGraph: steps run as a plain top-to-bottom asyncio pipeline. Each step is a
    stateless call to a fresh `pi -p ... --mode json` subprocess, so there is no shared
    graph state to thread through.
  - No vector store / embeddings: the "search the book" step instead spawns one `pi`
    subprocess per question with its working directory set to a scratch folder holding
    the book's full-text markdown, and only the `grep`/`find`/`read` tools enabled. The
    agent does its own thorough text search over the file instead of a similarity search
    over embedded chunks.
  - No structured-output API: pi has no schema-constrained output, so JSON-producing
    steps ask the model for raw JSON and repair it with a corrective follow-up call on
    parse failure.

A weak/free model can occasionally return a near-empty answer for a question under
concurrent load; answers under MIN_ANSWER_WORDS are retried with a fresh, reinforced
stateless call (up to MAX_THIN_ANSWER_ATTEMPTS times) rather than accepted as-is.

Every pi call is also capped at DEFAULT_PI_TIMEOUT_SECONDS (--pi-timeout /
PI_CALL_TIMEOUT_SECONDS): a stalled subprocess (rate limit, network stall, an agent
looping on tool calls) is killed and raised as a PiError rather than hanging forever,
which the same retry logic (_retry_step, STEP_RETRY_ATTEMPTS) then retries like any
other failure.

Every pi subprocess call is timed and its token usage (parsed from the `--mode json`
event stream) recorded into a RunProtocol, written out as `<book>_protocol.json`
alongside the other outputs -- runtime and input/output/total token counts, per call
and summed for the whole run.

Once metadata is extracted, the book is filed into the library convention
`books/Letter<X>/<Title, ':' -> '_'> - <Author>.<ext>` (X = first letter of the
title) unless --no-organize is passed; all derived outputs are written next to
wherever the book ends up.

Requires the `pi` CLI on PATH (or pass --pi-bin), plus: markitdown, pydantic, markdown,
weasyprint.
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import markdown as md_lib
from dotenv import load_dotenv
from markitdown import MarkItDown
from pydantic import BaseModel, Field

load_dotenv()

# Directory containing the books/Letter<X>/ library layout, i.e. the parent of
# the Letter<X> folders themselves (not the parent of a "books" folder).
DEFAULT_BOOKS_PATH = os.environ.get("BOOKS_PATH", "./books")

# Per-call wall-clock cap. Without this, a stalled pi subprocess (rate limit,
# network stall, an agent looping on tool calls without converging) hangs the
# whole run forever -- nothing ever raises, so even the retry logic never gets
# a chance to kick in. A timeout turns that into an ordinary PiError, which
# _retry_step already retries like any other failure.
DEFAULT_PI_TIMEOUT_SECONDS = float(os.environ.get("PI_CALL_TIMEOUT_SECONDS", "600"))

# Chunking for metadata extraction (front-of-book scan only; character-based, not
# token-based -- good enough since we only ever look at the first few chunks).
METADATA_CHUNK_SIZE = 4096 * 4
METADATA_CHUNK_OVERLAP_RATIO = 0.1
METADATA_MAX_CHUNKS_TO_SEARCH = 5

MAX_JSON_REPAIR_ATTEMPTS = 3
STEP_RETRY_ATTEMPTS = 3

# A weak/free model under concurrent load can occasionally return a near-empty
# answer for a question (observed: 1-113 output tokens instead of several hundred).
# Below this word count, the answer is treated as too thin and retried with a
# fresh, reinforced stateless call rather than accepted as-is.
MIN_ANSWER_WORDS = 100
MAX_THIN_ANSWER_ATTEMPTS = 2

QUESTIONS = [
    "What is this book fundamentally about? Lead with the most surprising or counterintuitive claim.",
    "What real-world problem motivated this book? Include concrete examples or case studies if available.",
    "What is the central claim or thesis? State it as a bold, testable proposition.",
    "What are the 3-5 key arguments supporting the thesis?",
    "How is the book organized and structured?",
    "What are the most important concepts or frameworks introduced?",
    "What are the 3 most vivid examples, case studies, or thought experiments used in the book?",
    "What evidence, examples, or case studies does the author emphasize?",
    "What are the main conclusions and recommendations?",
    "How does this work relate to other books or research in the field?",
    "What would change if the author's thesis is correct?",
    "What questions does the author leave unanswered?",
]


# ---------------------------------------------------------------------------
# Metadata model (unchanged from the LangGraph version)
# ---------------------------------------------------------------------------

class BookGenre(str, Enum):
    LITERARY_FICTION = "Literary-Fiction"
    SCIENCE_FICTION = "Science-Fiction"
    FANTASY = "Fantasy"
    MYSTERY = "Mystery"
    THRILLER = "Thriller"
    HORROR = "Horror"
    ROMANCE = "Romance"
    BIOGRAPHY = "Biography"
    MEMOIR = "Memoir"
    HISTORY = "History"
    PHILOSOPHY = "Philosophy"
    SCIENCE = "Science"
    PSYCHOLOGY = "Psychology"
    SELF_HELP = "Self-Help"
    TECHNICAL = "Technical"
    BUSINESS = "Business"
    POETRY = "Poetry"
    DRAMA = "Drama"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class BookMetadata(BaseModel):
    title: str = Field(..., description="Book title as it appears on the cover or title page")
    author: str = Field(..., description="Primary author name in 'FirstName LastName' format; 'Unknown' if not determinable")
    publication_year: int = Field(..., description="Year of first publication; 0 if unknown")
    genre: BookGenre = Field(..., description="Primary genre classification")
    custom_tags: List[str] = Field(default_factory=list, description="User-defined tags for categorization or themes")


class MetadataDecision(BaseModel):
    book_metadata: BookMetadata
    should_continue_search: bool = Field(
        ..., description="False if metadata is sufficient, True if more content should be scanned"
    )


# ---------------------------------------------------------------------------
# pi subprocess plumbing
# ---------------------------------------------------------------------------

class PiError(RuntimeError):
    pass


@dataclass
class PiCallRecord:
    """One completed pi subprocess call, for the protocol.json report."""

    label: str
    started_at: str
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: Optional[str]
    provider: Optional[str]


@dataclass
class RunProtocol:
    """Accumulates runtime and token usage across every pi call in a workflow run."""

    started_at: float = field(default_factory=time.time)
    calls: List[PiCallRecord] = field(default_factory=list)

    def record(self, call: PiCallRecord) -> None:
        self.calls.append(call)

    def to_dict(self, book_path: str) -> dict:
        finished_at = time.time()
        return {
            "book_path": book_path,
            "started_at": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "finished_at": datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat(),
            "duration_seconds": round(finished_at - self.started_at, 3),
            "pi_calls": len(self.calls),
            "tokens": {
                "input": sum(c.input_tokens for c in self.calls),
                "output": sum(c.output_tokens for c in self.calls),
                "total": sum(c.total_tokens for c in self.calls),
            },
            "cost_usd_total": round(sum(c.cost_usd for c in self.calls), 6),
            "calls": [asdict(c) for c in self.calls],
        }


def _build_pi_command(
    pi_bin: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tools: Optional[List[str]] = None,
    no_tools: bool = False,
) -> List[str]:
    cmd = [pi_bin, "--mode", "json", "--no-session", "-a", "-p", prompt]
    if provider:
        cmd += ["--provider", provider]
    if model:
        cmd += ["--model", model]
    if no_tools:
        cmd += ["--no-tools"]
    elif tools is not None:
        cmd += ["--tools", ",".join(tools)]
    return cmd


def _content_to_text(content) -> str:
    """Assistant message content may be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


async def run_pi(
    prompt: str,
    *,
    pi_bin: str = "pi",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tools: Optional[List[str]] = None,
    no_tools: bool = False,
    cwd: Optional[str] = None,
    protocol: Optional[RunProtocol] = None,
    label: str = "pi_call",
    timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> str:
    """Run one stateless pi turn and return the final assistant message text.

    Parses the `--mode json` event stream defensively: field names are taken from
    pi's docs, not a formal spec, so unknown/malformed lines are skipped rather than
    treated as fatal.
    """

    cmd = _build_pi_command(pi_bin, prompt, model=model, provider=provider, tools=tools, no_tools=no_tools)

    call_started_at = time.time()
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise PiError(f"pi call timed out after {timeout_seconds}s (label={label})")
    duration_seconds = time.time() - call_started_at

    if process.returncode != 0:
        raise PiError(f"pi exited with code {process.returncode}: {stderr.decode(errors='replace')[-2000:]}")

    last_assistant_text = ""
    last_usage: dict = {}
    for line in stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "message_end":
            message = event.get("message", {})
            if message.get("role") == "assistant":
                text = _content_to_text(message.get("content"))
                if text:
                    last_assistant_text = text
                if message.get("usage"):
                    last_usage = message["usage"]

    if not last_assistant_text:
        raise PiError(f"pi produced no assistant text. stderr: {stderr.decode(errors='replace')[-2000:]}")

    if protocol is not None:
        protocol.record(
            PiCallRecord(
                label=label,
                started_at=datetime.fromtimestamp(call_started_at, tz=timezone.utc).isoformat(),
                duration_seconds=round(duration_seconds, 3),
                input_tokens=last_usage.get("input", 0),
                output_tokens=last_usage.get("output", 0),
                total_tokens=last_usage.get("totalTokens", 0),
                cost_usd=last_usage.get("cost", {}).get("total", 0),
                model=model,
                provider=provider,
            )
        )

    return last_assistant_text


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n(.*)\n```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


async def run_pi_json(
    prompt: str,
    schema: dict,
    *,
    pi_bin: str = "pi",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    protocol: Optional[RunProtocol] = None,
    label: str = "pi_call_json",
    timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> dict:
    """Run a text-only pi turn and parse its reply as JSON, repairing on failure."""

    raw = await run_pi(
        prompt,
        pi_bin=pi_bin,
        model=model,
        provider=provider,
        no_tools=True,
        protocol=protocol,
        label=label,
        timeout_seconds=timeout_seconds,
    )

    last_error = None
    for attempt in range(MAX_JSON_REPAIR_ATTEMPTS):
        try:
            return json.loads(_strip_code_fences(raw))
        except json.JSONDecodeError as exc:
            last_error = exc
            repair_prompt = (
                "The following was supposed to be a single valid JSON object matching this schema:\n"
                f"{json.dumps(schema)}\n\n"
                f"It failed to parse with error: {exc}\n\n"
                "Here is the invalid output:\n"
                f"<output>\n{raw}\n</output>\n\n"
                "Return ONLY the corrected, valid JSON object. No markdown fences, no explanation."
            )
            raw = await run_pi(
                repair_prompt,
                pi_bin=pi_bin,
                model=model,
                provider=provider,
                no_tools=True,
                protocol=protocol,
                label=f"{label}_repair_{attempt + 1}",
                timeout_seconds=timeout_seconds,
            )

    raise PiError(f"Could not obtain valid JSON after {MAX_JSON_REPAIR_ATTEMPTS} repair attempts: {last_error}")


async def _retry_step(coro_fn, *args, attempts: int = STEP_RETRY_ATTEMPTS, **kwargs):
    last_exc = None
    for attempt in range(attempts):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, mirrors RetryPolicy
            last_exc = exc
    raise last_exc


# ---------------------------------------------------------------------------
# Step 1: load the book, converting PDF/EPUB to markdown via markitdown
# ---------------------------------------------------------------------------

def load_book_content(book_path: str) -> str:
    """Read/convert the book into markdown text, in memory only (no disk writes).

    Writing the .md file to disk is deferred until after the book has been filed
    into its library location (see organize_into_library), since that location
    depends on metadata extracted from this content.
    """
    if not os.path.exists(book_path):
        raise ValueError(f"File {book_path} does not exist.")

    extension = os.path.splitext(book_path)[1].lower()
    if extension == ".md":
        with open(book_path, "r", encoding="utf-8") as file:
            content = file.read()
    elif extension in (".epub", ".pdf"):
        content = MarkItDown(enable_plugins=False).convert(book_path).text_content
    else:
        raise ValueError(f"Unsupported file format: {book_path}")

    if not content.strip():
        raise ValueError(f"No content found in book {book_path}")

    return content


# ---------------------------------------------------------------------------
# Step 2: metadata extraction (front-of-book scan, iterative refine)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    overlap = int(chunk_size * overlap_ratio)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


async def extract_metadata(
    book_content: str,
    *,
    pi_bin: str,
    model: Optional[str],
    provider: Optional[str],
    protocol: RunProtocol,
    timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> BookMetadata:
    chunks = _chunk_text(book_content, METADATA_CHUNK_SIZE, METADATA_CHUNK_OVERLAP_RATIO)

    schema = MetadataDecision.model_json_schema()
    book_metadata: Optional[BookMetadata] = None

    for index, chunk in enumerate(chunks):
        if index > METADATA_MAX_CHUNKS_TO_SEARCH:
            raise ValueError(f"Could not extract metadata from first {index + 1} chunks of the book.")

        prompt_parts = [
            "You are a helpful assistant specialized in extracting book metadata from book contents.",
            "",
            "Instructions:",
            "- Focus on the title page, copyright page, or first few pages where metadata typically appears",
            "- For title: extract the full title including subtitle if present",
            "- For author: extract the primary author name (first listed if multiple)",
            "- For publication year: use the original publication year, not reprint/edition years",
            "- If any field cannot be determined with confidence, use 'Unknown' (or 0 for the year)",
            "- Set should_continue_search to false only when all fields are confidently extracted or clearly unavailable",
            "",
            f"Output schema: {json.dumps(schema)}",
            "",
            "Respond with ONLY a single JSON object matching the schema above. No markdown fences, no explanation.",
            "",
        ]
        if book_metadata is not None:
            prompt_parts += [
                f"Previously extracted metadata: {book_metadata.model_dump_json()}",
                "If this metadata is complete and accurate, set should_continue_search to false.",
                "If any field is 'Unknown' or uncertain, keep searching and improve it if this content helps.",
                "",
            ]
        prompt_parts += ["Content to analyze:", f"<content>\n{chunk}\n</content>"]
        prompt = "\n".join(prompt_parts)

        result = await run_pi_json(
            prompt,
            schema,
            pi_bin=pi_bin,
            model=model,
            provider=provider,
            protocol=protocol,
            label=f"metadata_chunk_{index}",
            timeout_seconds=timeout_seconds,
        )
        decision = MetadataDecision.model_validate(result)
        book_metadata = decision.book_metadata
        if not decision.should_continue_search:
            break

    return book_metadata


# ---------------------------------------------------------------------------
# Step 3: table of contents (parse markdown headers directly, then format via pi)
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _parse_headers(text: str) -> List[str]:
    outline = []
    for line in text.splitlines():
        match = _HEADER_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            outline.append(f"{'  ' * (level - 1)}- (H{level}) {title}")
    return outline


async def build_toc(
    book_content: str,
    *,
    pi_bin: str,
    model: Optional[str],
    provider: Optional[str],
    protocol: RunProtocol,
    timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> str:
    outline = _parse_headers(book_content)
    if not outline:
        return "_No markdown headers found in the source document._"

    prompt = "\n".join([
        "You are a helpful assistant specialized in creating tables of contents for books.",
        "",
        "Here is the hierarchical structure of the book based on markdown headers:",
        "",
        *outline,
        "",
        "Create a table of contents for the book based on this hierarchical structure.",
        "",
        "Instructions:",
        "- Preserve the hierarchical structure shown above",
        "- Use markdown formatting with appropriate heading levels or numbered lists",
        "- Keep it concise -- this is a structural outline, not a detailed description",
        "",
        "Reply with the table of contents only, without additional explanation.",
    ])

    return await run_pi(
        prompt,
        pi_bin=pi_bin,
        model=model,
        provider=provider,
        no_tools=True,
        protocol=protocol,
        label="toc",
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Step 4: question answering via thorough textsearch (grep/find/read over book.md)
# ---------------------------------------------------------------------------

async def answer_question(
    question: str,
    *,
    index: int,
    search_dir: str,
    pi_bin: str,
    model: Optional[str],
    provider: Optional[str],
    protocol: RunProtocol,
    timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> Dict[str, str]:
    system_and_task = "\n".join([
        "You are writing an engaging book summary for smart readers who want substance without academic stuffiness.",
        "The full text of the book is available in book.md in your working directory.",
        "Use the grep, find, and read tools to thoroughly search book.md for information relevant to the question below.",
        "Do multiple searches with different keywords if necessary before answering -- do not rely on a single grep.",
        "",
        "Writing guidelines:",
        "- Assume the reader is intelligent but unfamiliar with the specific field",
        "- Use active voice and direct statements",
        "- Explain technical terms in-line where they first appear (e.g., 'entropy--the measure of disorder')",
        "- For abstract or technical concepts, include at least one everyday analogy",
        "- Replace academic phrases ('the author posits', 'this section demonstrates') with direct statements",
        "",
        "Structure principles for the answer:",
        "1. Open with a concrete example, story, or provocative claim from the book when possible",
        "2. Build from concrete to conceptual -- give vivid examples before abstract frameworks",
        "3. Use analogies to clarify complex ideas",
        "4. Vary sentence structure: mix short punchy statements with longer explanatory ones",
        "5. End with why this matters or what it enables, when relevant",
        "",
        f"The answer shall be in markdown format, starting with `## {question}` as title.",
        "",
        f"Question to answer: {question}",
        "",
        "Provide a comprehensive and accurate answer based on what you find in book.md.",
    ])

    answer = await run_pi(
        system_and_task,
        pi_bin=pi_bin,
        model=model,
        provider=provider,
        tools=["grep", "find", "read"],
        cwd=search_dir,
        protocol=protocol,
        label=f"question_{index}",
        timeout_seconds=timeout_seconds,
    )

    for attempt in range(1, MAX_THIN_ANSWER_ATTEMPTS + 1):
        word_count = len(answer.split())
        if word_count >= MIN_ANSWER_WORDS:
            break
        reinforced_prompt = (
            f"{system_and_task}\n\n"
            f"Your previous attempt at this question was only {word_count} words -- far too thin. "
            "Use grep with several different keywords, read the relevant sections in full, and write "
            "a comprehensive, detailed, multi-paragraph answer this time."
        )
        answer = await run_pi(
            reinforced_prompt,
            pi_bin=pi_bin,
            model=model,
            provider=provider,
            tools=["grep", "find", "read"],
            cwd=search_dir,
            protocol=protocol,
            label=f"question_{index}_thin_retry_{attempt}",
            timeout_seconds=timeout_seconds,
        )

    return {"question": question, "answer": answer}


# ---------------------------------------------------------------------------
# Step 5 & 6: long summary and short summary
# ---------------------------------------------------------------------------

async def write_long_summary(
    metadata: BookMetadata,
    toc: str,
    qa_pairs: List[Dict[str, str]],
    *,
    pi_bin: str,
    model: Optional[str],
    provider: Optional[str],
    protocol: RunProtocol,
    timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> str:
    prompt_parts = [
        "You are a helpful assistant specialized in creating comprehensive book summaries based on question-answer pairs.",
        "",
        "Using the following information, create a comprehensive book summary.",
        f"Book Metadata:\n<metadata>\n{metadata.model_dump_json()}\n</metadata>",
        f"Table of Contents:\n<table_of_contents>\n{toc}\n</table_of_contents>",
        "Q&A Pairs:\n<questions_and_answers>",
    ]
    for qa in qa_pairs:
        prompt_parts += [f"<question>\n{qa['question']}\n</question>", f"<answer>\n{qa['answer']}\n</answer>"]
    prompt_parts += [
        "</questions_and_answers>",
        "",
        "Create a compelling book summary that balances intellectual rigor with readability.",
        "",
        "Length target: 2000-3000 words (approximately 7-10 minute read)",
        "- Introduction with metadata: ~200 words",
        "- Main content sections: ~2400 words",
        "- Conclusion: ~200 words",
        "",
        "IMPORTANT: Do NOT include word counts in the output. The length targets above are internal guidance only.",
        "",
        "Style principles:",
        "- Write for a curious generalist, not a specialist",
        "- Use active voice and concrete examples",
        "- Explain WHY each concept matters before diving into WHAT it is",
        "- Replace academic phrases with direct statements",
        "- Break up dense paragraphs with subheadings and shorter sections where appropriate",
        "- Include vivid examples or thought experiments for major concepts",
        "- End key sections with a 'so what' takeaway when relevant",
        "- Define technical terms in-line",
        "- Vary sentence length: mix punchy statements with longer explanations",
        "",
        "Structure:",
        "- Synthesize the Q&A pairs into a coherent narrative",
        "- Use the table of contents to guide the overall organization",
        "- Create clear, descriptive, engaging section headings",
        "- Ensure smooth transitions between sections -- add a brief bridge sentence at the start of each section",
        "- Include the book metadata naturally in the introduction",
        "",
        "Provide the complete summary in markdown format.",
    ]
    return await run_pi(
        "\n".join(prompt_parts),
        pi_bin=pi_bin,
        model=model,
        provider=provider,
        no_tools=True,
        protocol=protocol,
        label="summary",
        timeout_seconds=timeout_seconds,
    )


async def write_short_summary(
    metadata: BookMetadata,
    full_summary: str,
    *,
    pi_bin: str,
    model: Optional[str],
    provider: Optional[str],
    protocol: RunProtocol,
    timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> str:
    prompt = "\n".join([
        "You are a helpful assistant specialized in creating concise book summaries.",
        "",
        "Using the following book metadata and full summary, create a concise short summary.",
        f"Book Metadata:\n<metadata>\n{metadata.model_dump_json()}\n</metadata>",
        f"Full Summary:\n<full_summary>\n{full_summary}\n</full_summary>",
        "",
        "Instructions:",
        "- Use Title, Author, Publication Year, and Genre from the metadata to introduce the book in the first heading",
        "- Length target: 200-300 words (approximately 1-2 minute read)",
        "- Write for a curious generalist",
        "- Use active voice and concrete examples",
        "- Explain WHY each concept matters before diving into WHAT it is",
        "- Replace academic phrases with direct statements",
        "- Define technical terms in-line",
        "- Vary sentence length: mix punchy statements with longer explanations",
        "",
        "Provide the complete short summary in markdown format. Answer with only the summary, no explanation.",
    ])
    return await run_pi(
        prompt,
        pi_bin=pi_bin,
        model=model,
        provider=provider,
        no_tools=True,
        protocol=protocol,
        label="short_summary",
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# File path helpers (unchanged from the LangGraph version)
# ---------------------------------------------------------------------------

def book_to_md_path(book_path: str) -> str:
    return book_path.replace(os.path.splitext(book_path)[1], ".md")


def book_to_metadata_json_path(book_path: str) -> str:
    return book_path.replace(os.path.splitext(book_path)[1], "_metadata.json")


def book_to_summary_md_path(book_path: str) -> str:
    return book_path.replace(os.path.splitext(book_path)[1], "_summary.md")


def book_to_shortsummary_md_path(book_path: str) -> str:
    return book_path.replace(os.path.splitext(book_path)[1], "_shortsummary.md")


def book_to_summary_pdf_path(book_path: str) -> str:
    return book_path.replace(os.path.splitext(book_path)[1], "_summary.pdf")


def book_to_shortsummary_pdf_path(book_path: str) -> str:
    return book_path.replace(os.path.splitext(book_path)[1], "_shortsummary.pdf")


def book_to_protocol_path(book_path: str) -> str:
    return book_path.replace(os.path.splitext(book_path)[1], "_protocol.json")


# ---------------------------------------------------------------------------
# Library convention: books/Letter<X>/<Title, ':' -> '_'> - <Author>.<ext>
# ---------------------------------------------------------------------------

_FILENAME_UNSAFE_RE = re.compile(r'[\\/:*?"<>|]')


def _sanitize_for_filename(text: str) -> str:
    """Replace filesystem-unsafe characters with '_', matching the convention's
    demonstrated 'Title: Subtitle' -> 'Title_ Subtitle' colon handling, extended to
    the rest of the reserved characters so a title/author can't break path construction."""
    return _FILENAME_UNSAFE_RE.sub("_", text).strip()


def library_letter_for_title(title: str) -> str:
    stripped = title.strip()
    first_char = stripped[0].upper() if stripped else ""
    return first_char if first_char.isalpha() else "#"


def library_filename(title: str, author: str, ext: str) -> str:
    return f"{_sanitize_for_filename(title)} - {_sanitize_for_filename(author)}{ext}"


def library_path_for_book(books_path: str, title: str, author: str, ext: str) -> str:
    letter = library_letter_for_title(title)
    return os.path.join(books_path, f"Letter{letter}", library_filename(title, author, ext))


def organize_into_library(book_path: str, metadata: BookMetadata, books_path: str) -> str:
    """Move the source book file to its Letter<X>/ location under books_path if it
    isn't there already, and return the (possibly new) path. All derived outputs
    are written next to whatever path this returns."""
    ext = os.path.splitext(book_path)[1]
    target_path = library_path_for_book(books_path, metadata.title, metadata.author, ext)

    if os.path.abspath(target_path) == os.path.abspath(book_path):
        return book_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"Filing book into library: {target_path}")
    shutil.move(book_path, target_path)
    return target_path


def is_complete(book_path: str) -> bool:
    return all(
        os.path.exists(path(book_path))
        for path in (
            book_to_md_path,
            book_to_metadata_json_path,
            book_to_summary_md_path,
            book_to_shortsummary_md_path,
            book_to_summary_pdf_path,
            book_to_shortsummary_pdf_path,
            book_to_protocol_path,
        )
    )


def _ensure_macos_homebrew_lib_path() -> None:
    """weasyprint's cffi bindings need Pango/GObject .dylibs; on macOS, Homebrew
    installs them under a prefix the dynamic linker doesn't search by default."""
    if sys.platform != "darwin" or "DYLD_LIBRARY_PATH" in os.environ:
        return
    for prefix in ("/opt/homebrew", "/usr/local"):
        lib_dir = os.path.join(prefix, "lib")
        if os.path.isdir(lib_dir):
            os.environ["DYLD_LIBRARY_PATH"] = lib_dir
            return


def write_pdf(markdown_content: str, pdf_path: str) -> None:
    _ensure_macos_homebrew_lib_path()
    from weasyprint import HTML  # imported lazily: needs native Pango/GObject libs

    html_content = md_lib.markdown(markdown_content, extensions=["extra", "codehilite", "toc"])
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Georgia', 'Times New Roman', serif; line-height: 1.6;
                     max-width: 800px; margin: 40px auto; padding: 20px; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 30px; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 25px; }}
            h3 {{ color: #7f8c8d; margin-top: 20px; }}
            p {{ margin: 15px 0; text-align: justify; }}
            code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}
            pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            blockquote {{ border-left: 4px solid #3498db; padding-left: 20px; margin: 20px 0; color: #555; font-style: italic; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
            th {{ background-color: #2c3e50; color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f4f4f4; }}
            @page {{ margin: 2cm; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    HTML(string=styled_html).write_pdf(pdf_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_book_workflow(
    book_path: str,
    *,
    pi_bin: str = "pi",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    force: bool = False,
    organize: bool = True,
    books_path: str = DEFAULT_BOOKS_PATH,
    pi_timeout_seconds: float = DEFAULT_PI_TIMEOUT_SECONDS,
) -> None:
    if not force and is_complete(book_path):
        print(f"All outputs already exist for {book_path}, skipping. Use --force to regenerate.")
        return

    protocol = RunProtocol()
    try:
        print(f"Loading and converting {book_path} ...")
        book_content = load_book_content(book_path)

        print("Extracting book metadata ...")
        metadata = await _retry_step(
            extract_metadata,
            book_content,
            pi_bin=pi_bin,
            model=model,
            provider=provider,
            protocol=protocol,
            timeout_seconds=pi_timeout_seconds,
        )
        print(f"  {metadata.title} by {metadata.author} ({metadata.publication_year}, {metadata.genre.value})")

        if organize:
            book_path = organize_into_library(book_path, metadata, books_path)

        with open(book_to_md_path(book_path), "w", encoding="utf-8") as file:
            file.write(book_content)
        with open(book_to_metadata_json_path(book_path), "w", encoding="utf-8") as file:
            file.write(metadata.model_dump_json(indent=4))

        print("Building table of contents ...")
        toc = await _retry_step(
            build_toc,
            book_content,
            pi_bin=pi_bin,
            model=model,
            provider=provider,
            protocol=protocol,
            timeout_seconds=pi_timeout_seconds,
        )

        print(f"Answering {len(QUESTIONS)} questions via textsearch over the book ...")
        search_dir = os.path.join(
            os.path.dirname(os.path.abspath(book_path)), f".{os.path.basename(book_to_md_path(book_path))}.search"
        )
        os.makedirs(search_dir, exist_ok=True)
        search_book_path = os.path.join(search_dir, "book.md")
        with open(search_book_path, "w", encoding="utf-8") as file:
            file.write(book_content)

        try:
            qa_pairs = await asyncio.gather(
                *(
                    _retry_step(
                        answer_question,
                        question,
                        index=index,
                        search_dir=search_dir,
                        pi_bin=pi_bin,
                        model=model,
                        provider=provider,
                        protocol=protocol,
                        timeout_seconds=pi_timeout_seconds,
                    )
                    for index, question in enumerate(QUESTIONS)
                )
            )
        finally:
            os.remove(search_book_path)
            os.rmdir(search_dir)

        print("Writing full summary ...")
        summary = await _retry_step(
            write_long_summary,
            metadata,
            toc,
            list(qa_pairs),
            pi_bin=pi_bin,
            model=model,
            provider=provider,
            protocol=protocol,
            timeout_seconds=pi_timeout_seconds,
        )
        with open(book_to_summary_md_path(book_path), "w", encoding="utf-8") as file:
            file.write(summary)
        write_pdf(summary, book_to_summary_pdf_path(book_path))

        print("Writing short summary ...")
        short_summary = await _retry_step(
            write_short_summary,
            metadata,
            summary,
            pi_bin=pi_bin,
            model=model,
            provider=provider,
            protocol=protocol,
            timeout_seconds=pi_timeout_seconds,
        )
        with open(book_to_shortsummary_md_path(book_path), "w", encoding="utf-8") as file:
            file.write(short_summary)
        write_pdf(short_summary, book_to_shortsummary_pdf_path(book_path))
    finally:
        protocol_dict = protocol.to_dict(book_path)
        with open(book_to_protocol_path(book_path), "w", encoding="utf-8") as file:
            json.dump(protocol_dict, file, indent=4)

    if not is_complete(book_path):
        raise RuntimeError("Workflow finished but not all expected output files were created.")

    tokens = protocol_dict["tokens"]
    print(
        f"Done in {protocol_dict['duration_seconds']:.1f}s -- "
        f"{protocol_dict['pi_calls']} pi calls, "
        f"tokens in={tokens['input']} out={tokens['output']} total={tokens['total']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a book using the pi coding agent.")
    parser.add_argument("book_path", help="Path to the book file (.md, .pdf, or .epub)")
    parser.add_argument("--pi-bin", default="pi", help="Path to the pi executable (default: pi)")
    parser.add_argument("--model", default=None, help="Model pattern/ID to pass to pi (e.g. sonnet:high)")
    parser.add_argument("--provider", default=None, help="Provider to pass to pi (e.g. anthropic, openai, google)")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if they already exist")
    parser.add_argument(
        "--books-path",
        default=DEFAULT_BOOKS_PATH,
        help="Directory containing the Letter<X>/ library layout "
        "(default: $BOOKS_PATH env var if set, else ./books)",
    )
    parser.add_argument(
        "--no-organize",
        action="store_true",
        help="Don't file the book into <books-path>/Letter<X>/<Title> - <Author>.<ext>; write outputs next to book_path as given",
    )
    parser.add_argument(
        "--pi-timeout",
        type=float,
        default=DEFAULT_PI_TIMEOUT_SECONDS,
        help="Per-call timeout in seconds before a pi subprocess is killed and retried "
        "(default: $PI_CALL_TIMEOUT_SECONDS env var if set, else 600)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            run_book_workflow(
                args.book_path,
                pi_bin=args.pi_bin,
                model=args.model,
                provider=args.provider,
                force=args.force,
                organize=not args.no_organize,
                books_path=args.books_path,
                pi_timeout_seconds=args.pi_timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
