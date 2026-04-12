"""Generate structured medical content from a topic string (no PDF needed).

Uses GPT to create a MedicalContent object as if it were extracted from a
BoardBuddy-format PDF, so the rest of the pipeline works identically.
"""

from __future__ import annotations

import json
import os

from tools.models import AnswerChoice, MedicalContent, WrongAnswerExplanation
from tools.provider import chat_json, get_text_model_name

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", get_text_model_name())


def generate_content_from_topic(topic: str) -> MedicalContent:
    """Generate structured medical education content from a topic string.

    The LLM creates a complete teaching case: clinical vignette, MCQ,
    pathophysiology, differentials, and educational objective — as if
    extracted from a real study resource.
    """
    prompt = f"""You are a medical education content creator. Given a topic, create a complete
teaching case in the style of a BoardBuddy study diagram.

**Topic:** {topic}

Generate a JSON object with ALL of these fields populated with accurate, detailed medical content:

{{
  "topic": "{topic}",
  "subject": "<medical subject, e.g. Pathophysiology, Pharmacology>",
  "system": "<organ system, e.g. Cardiovascular, Musculoskeletal>",
  "clinical_vignette": "<A 3-5 sentence patient presentation that illustrates the topic. Include age, sex, symptoms, relevant history, and physical exam findings.>",
  "question_stem": "<A 'Which of the following...' style exam question>",
  "answer_choices": [
    {{"letter": "A", "text": "<plausible wrong answer>"}},
    {{"letter": "B", "text": "<plausible wrong answer>"}},
    {{"letter": "C", "text": "<correct answer>"}},
    {{"letter": "D", "text": "<plausible wrong answer>"}},
    {{"letter": "E", "text": "<plausible wrong answer>"}}
  ],
  "correct_answer": "<the correct answer text>",
  "correct_answer_letter": "<letter>",
  "pathophysiology": "<Detailed 4-8 sentence explanation of the underlying mechanism. This is the CORE teaching content — be thorough, step-by-step.>",
  "key_info": "<2-3 key clinical facts or pearls>",
  "why_section": "<Why this mechanism matters clinically — connect pathophys to symptoms>",
  "explanation": "<Full explanation of why the correct answer is right>",
  "wrong_answer_explanations": [
    {{"letter": "A", "text": "<answer text>", "explanation": "<why it's wrong>"}},
    {{"letter": "B", "text": "<answer text>", "explanation": "<why it's wrong>"}},
    {{"letter": "D", "text": "<answer text>", "explanation": "<why it's wrong>"}},
    {{"letter": "E", "text": "<answer text>", "explanation": "<why it's wrong>"}}
  ],
  "educational_objective": "<One sentence: what the student should understand after watching>",
  "bottom_line": "<One sentence clinical takeaway>",
  "diagram_description": "<Description of an ideal diagram/illustration for this topic>",
  "diagram_labels": ["label1", "label2", "label3"]
}}

Make the pathophysiology section especially detailed — it drives the video's mechanism scenes.
Return ONLY valid JSON."""

    data = chat_json(
        "You are a medical education content creator.",
        prompt,
        model=DEFAULT_MODEL,
        max_tokens=4096,
        temperature=0.3,
    )

    return MedicalContent(
        topic=data.get("topic", topic),
        subject=data.get("subject", ""),
        system=data.get("system", ""),
        clinical_vignette=data.get("clinical_vignette", ""),
        question_stem=data.get("question_stem", ""),
        answer_choices=[
            AnswerChoice(**c) for c in data.get("answer_choices", [])
        ],
        correct_answer=data.get("correct_answer", ""),
        correct_answer_letter=data.get("correct_answer_letter", ""),
        pathophysiology=data.get("pathophysiology", ""),
        key_info=data.get("key_info", ""),
        why_section=data.get("why_section", ""),
        explanation=data.get("explanation", ""),
        wrong_answer_explanations=[
            WrongAnswerExplanation(**w)
            for w in data.get("wrong_answer_explanations", [])
        ],
        educational_objective=data.get("educational_objective", ""),
        bottom_line=data.get("bottom_line", ""),
        diagram_description=data.get("diagram_description", ""),
        diagram_labels=data.get("diagram_labels", []),
    )
