"""Extract and classify content from medical education PDFs (BoardBuddy format)."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from tools.models import AnswerChoice, MedicalContent, WrongAnswerExplanation


def extract_pdf(pdf_path: str | Path) -> MedicalContent:
    """Extract structured medical content from a BoardBuddy-format PDF."""
    doc = fitz.open(str(pdf_path))
    pages_text = [page.get_text() for page in doc]
    full_text = "\n".join(pages_text)
    doc.close()

    return MedicalContent(
        topic=_extract_topic(pages_text[0]) if pages_text else "",
        subject=_extract_field(pages_text[0], "SUBJECT") if pages_text else "",
        system=_extract_field(pages_text[0], "SYSTEM") if pages_text else "",
        clinical_vignette=_extract_vignette(full_text),
        question_stem=_extract_question_stem(full_text),
        answer_choices=_extract_answer_choices(full_text),
        correct_answer=_extract_correct_answer(full_text),
        correct_answer_letter=_extract_correct_answer_letter(full_text),
        diagram_description=_extract_section(full_text, "Image Prompt:"),
        diagram_labels=_extract_labels(full_text),
        bottom_line=_extract_section(full_text, "Bottom Line:"),
        pathophysiology=_extract_section(full_text, "Pathophysiology:"),
        key_info=_extract_section(full_text, "Key Info:"),
        why_section=_extract_section(full_text, "Why:"),
        explanation=_extract_section(full_text, "Explanation:"),
        wrong_answer_explanations=_extract_wrong_answers(full_text),
        educational_objective=_extract_educational_objective(full_text),
    )


def _extract_topic(page1_text: str) -> str:
    """Extract topic from first page — it's the line after 'Diagram'."""
    lines = [l.strip() for l in page1_text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if line == "Diagram" and i + 1 < len(lines):
            return lines[i + 1]
    # Fallback: look for text between Study Diagram and SUBJECT
    m = re.search(r"Diagram\n(.+?)\nSUBJECT", page1_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_field(page1_text: str, field_name: str) -> str:
    """Extract a labeled field from page 1 (e.g., SUBJECT -> Pathophysiology)."""
    lines = [l.strip() for l in page1_text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if line == field_name and i + 1 < len(lines):
            return lines[i + 1]
    return ""


def _extract_vignette(text: str) -> str:
    """Extract the clinical vignette (from 'Question' to the MCQ options)."""
    m = re.search(
        r"Question\n(.+?)(?=\n[A-F]\.\s)",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def _extract_question_stem(text: str) -> str:
    """Extract just the question stem (the 'Which of the following...' part)."""
    vignette = _extract_vignette(text)
    # The question is usually the last sentence starting with "Which"
    sentences = re.split(r"(?<=[.?])\s+", vignette)
    for s in reversed(sentences):
        if "which of the following" in s.lower() or "what is" in s.lower():
            return s.strip()
    return ""


def _extract_answer_choices(text: str) -> list[AnswerChoice]:
    """Extract MCQ answer choices (A through F)."""
    choices = []
    pattern = r"^([A-F])\.\s+(.+)$"
    for m in re.finditer(pattern, text, re.MULTILINE):
        choices.append(AnswerChoice(letter=m.group(1), text=m.group(2).strip()))
    return choices


def _extract_correct_answer(text: str) -> str:
    """Extract the correct answer text."""
    m = re.search(r"Correct Answer:\s*[A-F]\.\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _extract_correct_answer_letter(text: str) -> str:
    """Extract just the correct answer letter."""
    m = re.search(r"Correct Answer:\s*([A-F])\.", text)
    return m.group(1) if m else ""


def _extract_section(text: str, header: str) -> str:
    """Extract a named section's content until the next section header."""
    # Section headers we know about
    headers = [
        "Image Prompt:",
        "Bottom Line:",
        "Pathophysiology:",
        "Key Info:",
        "Why:",
        "Explanation:",
        "Educational objective:",
        "LABELS:",
        "TITLE:",
    ]
    escaped = re.escape(header)
    # Find start of this section
    m = re.search(escaped + r"\s*\n?", text)
    if not m:
        return ""
    start = m.end()

    # Find the next section header after this one
    next_pos = len(text)
    for h in headers:
        if h == header:
            continue
        hm = re.search(re.escape(h), text[start:])
        if hm and hm.start() + start < next_pos:
            next_pos = hm.start() + start

    content = text[start:next_pos].strip()
    # Clean up multi-line text
    content = re.sub(r"\n\s*\n", "\n\n", content)
    return content


def _extract_labels(text: str) -> list[str]:
    """Extract diagram labels from LABELS: line."""
    m = re.search(r"LABELS?:\s*(.+?)(?:\n|$)", text)
    if m:
        return [l.strip() for l in m.group(1).split(",") if l.strip()]
    return []


def _extract_wrong_answers(text: str) -> list[WrongAnswerExplanation]:
    """Extract wrong answer explanations from the Explanation section."""
    explanations = []
    # Pattern: (Choice X) or (Choices X and Y)
    pattern = r"\(Choice(?:s)?\s+([A-F](?:\s+and\s+[A-F])?)\)\s+(.+?)(?=\(Choice|\nEducational objective:|$)"
    for m in re.finditer(pattern, text, re.DOTALL):
        letters = m.group(1)
        explanation = m.group(2).strip()
        # Handle "Choices A and E" case
        for letter in re.findall(r"[A-F]", letters):
            # Find the original answer choice text
            choice_text = ""
            cm = re.search(rf"^{letter}\.\s+(.+)$", text, re.MULTILINE)
            if cm:
                choice_text = cm.group(1).strip()
            explanations.append(
                WrongAnswerExplanation(
                    letter=letter,
                    text=choice_text,
                    explanation=explanation,
                )
            )
    return explanations


def _extract_educational_objective(text: str) -> str:
    """Extract the educational objective."""
    m = re.search(r"Educational objective:\s*(.+?)$", text, re.DOTALL)
    return m.group(1).strip() if m else ""
