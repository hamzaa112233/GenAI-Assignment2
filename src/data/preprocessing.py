"""
Text preprocessing utilities for the clinical trial eligibility NLP project.

Provides functions for:
- Numbering sentences in patient descriptions
- Formatting input prompts for instruction-tuned models
- Parsing structured model output into predictions
- Truncating long patient descriptions
- Cleaning and normalizing text
"""

import json
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# System prompt for the clinical trial eligibility task
SYSTEM_PROMPT = (
    "You are a clinical trial eligibility assessment system. Given a patient "
    "description with numbered sentences, a criterion type (inclusion or "
    "exclusion), and the criterion text, determine the patient's eligibility "
    "status and identify the supporting sentences from the patient description."
)

# Maximum number of characters for patient description before truncation
MAX_NOTE_CHARS = 4000

# Valid eligibility labels
VALID_INCLUSION_LABELS = {"included", "not included", "not enough information", "not applicable"}
VALID_EXCLUSION_LABELS = {"excluded", "not excluded", "not enough information", "not applicable"}
ALL_VALID_LABELS = VALID_INCLUSION_LABELS | VALID_EXCLUSION_LABELS


# ---------------------------------------------------------------------------
# Sentence numbering
# ---------------------------------------------------------------------------


def number_sentences(text: str) -> str:
    """Number each sentence in a text, prefixing with ``N.`` format.

    If the text already has numbered sentences (lines starting with a digit
    followed by a period), it is returned unchanged.  Otherwise, each line
    or sentence-split segment is numbered sequentially starting from 0.

    Parameters
    ----------
    text : str
        Raw patient description text.

    Returns
    -------
    str
        Text with each sentence numbered ``0. ...\n1. ...\n`` etc.

    Examples
    --------
    >>> number_sentences("The patient is 45 years old.\\nHe has diabetes.")
    '0. The patient is 45 years old.\\n1. He has diabetes.'
    """
    if not text or not text.strip():
        return text

    # Check if already numbered (lines starting with "N. ")
    lines = text.strip().split("\n")
    if lines and re.match(r"^\d+\.\s", lines[0]):
        return text

    # Split on newlines (common in clinical notes)
    numbered_lines = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if line:
            numbered_lines.append(f"{idx}. {line}")

    return "\n".join(numbered_lines)


def extract_sentence_by_id(note: str, sentence_id: int) -> Optional[str]:
    """Extract a specific sentence from a numbered patient note.

    Parameters
    ----------
    note : str
        The numbered patient note.
    sentence_id : int
        The 0-based sentence index to extract.

    Returns
    -------
    Optional[str]
        The sentence text, or ``None`` if the ID is not found.
    """
    pattern = rf"^{sentence_id}\.\s+(.+)$"
    match = re.search(pattern, note, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def count_sentences(note: str) -> int:
    """Count the number of numbered sentences in a patient note.

    Parameters
    ----------
    note : str
        The numbered patient note.

    Returns
    -------
    int
        Number of sentences found.
    """
    return len(re.findall(r"^\d+\.\s", note, re.MULTILINE))


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize text by collapsing whitespace and stripping.

    - Replaces multiple spaces/tabs with a single space
    - Strips leading/trailing whitespace
    - Preserves newlines (important for numbered sentences)

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text.
    """
    if not text:
        return ""

    # Collapse horizontal whitespace (spaces, tabs) but preserve newlines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Collapse multiple spaces/tabs into one
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()
        cleaned_lines.append(line)

    # Remove empty trailing lines
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def truncate_note(note: str, max_chars: int = MAX_NOTE_CHARS) -> str:
    """Truncate a patient note to a maximum character length.

    Truncation is done at sentence boundaries to avoid cutting mid-sentence.
    A truncation notice is appended if the note was shortened.

    Parameters
    ----------
    note : str
        The numbered patient note.
    max_chars : int
        Maximum number of characters (default 4000).

    Returns
    -------
    str
        Truncated note (or original if within limit).
    """
    if len(note) <= max_chars:
        return note

    lines = note.split("\n")
    truncated = []
    current_length = 0

    for line in lines:
        # +1 for the newline character
        if current_length + len(line) + 1 > max_chars:
            break
        truncated.append(line)
        current_length += len(line) + 1

    if len(truncated) < len(lines):
        truncated.append("[Note truncated for length]")

    return "\n".join(truncated)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_input_prompt(
    note: str,
    criterion_type: str,
    criterion_text: str,
    include_system: bool = True,
) -> str:
    """Format a single sample into an instruction-tuning input prompt.

    The prompt structure follows a clear separation between the patient
    description, the criterion to evaluate, and the expected output format.

    Parameters
    ----------
    note : str
        The numbered patient description.
    criterion_type : str
        Either ``'inclusion'`` or ``'exclusion'``.
    criterion_text : str
        The criterion text to evaluate against the patient.
    include_system : bool
        Whether to include the system prompt prefix.

    Returns
    -------
    str
        The formatted input prompt.
    """
    note = clean_text(note)
    note = truncate_note(note)
    criterion_text = clean_text(criterion_text)

    parts = []

    if include_system:
        parts.append(f"### System:\n{SYSTEM_PROMPT}\n")

    parts.append(f"### Patient Description:\n{note}\n")
    parts.append(f"### Criterion Type: {criterion_type}\n")
    parts.append(f"### Criterion Text:\n{criterion_text}\n")
    parts.append(
        "### Task:\n"
        "Determine the patient's eligibility for this criterion. Provide:\n"
        "1. The eligibility label\n"
        "2. The sentence IDs from the patient description that support your decision\n"
    )
    parts.append("### Response:\n")

    return "\n".join(parts)


def format_target_output(
    eligibility: str,
    evidence_sentences: str,
) -> str:
    """Format the ground-truth target as a structured output string.

    Parameters
    ----------
    eligibility : str
        The eligibility label (e.g. ``'included'``, ``'not excluded'``).
    evidence_sentences : str
        String representation of sentence IDs (e.g. ``'[0, 1, 2]'``).

    Returns
    -------
    str
        Structured JSON output string.
    """
    # Parse sentence IDs from string representation
    sentence_ids = parse_sentence_ids(evidence_sentences)

    output = {
        "eligibility": eligibility.strip().lower(),
        "supporting_sentences": sentence_ids,
    }

    return json.dumps(output, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_sentence_ids(sentence_str: str) -> List[int]:
    """Parse sentence IDs from a string like ``'[0, 1, 2]'`` or ``'[]'``.

    Parameters
    ----------
    sentence_str : str
        String representation of sentence ID list.

    Returns
    -------
    List[int]
        Parsed list of integer sentence IDs.
    """
    if not sentence_str or sentence_str.strip() in ("[]", ""):
        return []

    # Try JSON parsing first
    try:
        parsed = json.loads(sentence_str)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract all integers from the string
    ids = re.findall(r"\d+", sentence_str)
    return [int(x) for x in ids]


def parse_model_output(output_text: str) -> Dict:
    """Parse structured model output into eligibility prediction and evidence.

    Handles both clean JSON output and free-text output from the model.

    Parameters
    ----------
    output_text : str
        The raw model output text.

    Returns
    -------
    Dict
        Dictionary with keys:
        - ``'eligibility'``: The predicted eligibility label (str).
        - ``'supporting_sentences'``: List of supporting sentence IDs (List[int]).
        - ``'raw_output'``: The original output text (str).
        - ``'parse_success'``: Whether parsing was successful (bool).
    """
    result = {
        "eligibility": "unknown",
        "supporting_sentences": [],
        "raw_output": output_text.strip(),
        "parse_success": False,
    }

    if not output_text or not output_text.strip():
        return result

    text = output_text.strip()

    # Strategy 1: Try to parse as JSON directly
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if "eligibility" in parsed:
                result["eligibility"] = parsed["eligibility"].strip().lower()
            if "supporting_sentences" in parsed:
                result["supporting_sentences"] = [
                    int(x) for x in parsed["supporting_sentences"]
                ]
            result["parse_success"] = True
            return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 2: Find JSON object within the text
    json_match = re.search(r"\{[^{}]+\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                if "eligibility" in parsed:
                    result["eligibility"] = parsed["eligibility"].strip().lower()
                if "supporting_sentences" in parsed:
                    result["supporting_sentences"] = [
                        int(x) for x in parsed["supporting_sentences"]
                    ]
                result["parse_success"] = True
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: Look for eligibility label in text
    text_lower = text.lower()
    for label in sorted(ALL_VALID_LABELS, key=len, reverse=True):
        if label in text_lower:
            result["eligibility"] = label
            result["parse_success"] = True
            break

    # Look for sentence IDs in text
    sentence_match = re.search(
        r"(?:supporting[_ ]sentences|sentences?|evidence)\s*[:=]\s*\[([^\]]*)\]",
        text_lower,
    )
    if sentence_match:
        ids_str = sentence_match.group(1)
        result["supporting_sentences"] = [int(x) for x in re.findall(r"\d+", ids_str)]

    return result


def validate_prediction(
    prediction: Dict,
    criterion_type: str,
) -> Dict:
    """Validate a parsed prediction and flag any issues.

    Parameters
    ----------
    prediction : Dict
        Output from :func:`parse_model_output`.
    criterion_type : str
        The criterion type (``'inclusion'`` or ``'exclusion'``).

    Returns
    -------
    Dict
        The prediction dict with an additional ``'valid'`` boolean and
        ``'issues'`` list.
    """
    issues = []
    label = prediction.get("eligibility", "unknown")

    if criterion_type == "inclusion":
        valid_labels = VALID_INCLUSION_LABELS
    elif criterion_type == "exclusion":
        valid_labels = VALID_EXCLUSION_LABELS
    else:
        valid_labels = ALL_VALID_LABELS
        issues.append(f"Unknown criterion_type: {criterion_type}")

    if label not in valid_labels:
        issues.append(
            f"Invalid label '{label}' for {criterion_type} criterion. "
            f"Valid labels: {sorted(valid_labels)}"
        )

    if not prediction.get("parse_success", False):
        issues.append("Model output could not be parsed successfully")

    prediction["valid"] = len(issues) == 0
    prediction["issues"] = issues
    return prediction


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------


def batch_parse_outputs(
    outputs: List[str],
    criterion_types: List[str],
) -> List[Dict]:
    """Parse and validate a batch of model outputs.

    Parameters
    ----------
    outputs : List[str]
        List of raw model output strings.
    criterion_types : List[str]
        Corresponding criterion types.

    Returns
    -------
    List[Dict]
        List of validated prediction dictionaries.
    """
    assert len(outputs) == len(criterion_types), (
        f"Mismatch: {len(outputs)} outputs vs {len(criterion_types)} criterion types"
    )

    results = []
    for output, ctype in zip(outputs, criterion_types):
        parsed = parse_model_output(output)
        validated = validate_prediction(parsed, ctype)
        results.append(validated)

    return results
