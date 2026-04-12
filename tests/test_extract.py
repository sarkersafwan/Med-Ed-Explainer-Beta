from pathlib import Path

from tools.extract import extract_pdf


def test_extract_pdf_boardbuddy_fixture():
    content = extract_pdf(Path('input.pdf'))

    assert content.topic
    assert content.clinical_vignette
    assert content.pathophysiology
    assert content.educational_objective
    assert len(content.answer_choices) >= 4
