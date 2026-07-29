from pathlib import Path
import re


FORBIDDEN_UNSUPPORTED_CLAIMS = [
    "80,000 manually labeled",
    "F1 score of 0.847",
    "direct causality",
    "statistically distinct ingredient preferences",
    "10.1109/ACCESS.2024.DOI",
]


def _manuscript_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "manuscript"
        / "revised"
        / "main (1).tex"
    )


def test_revised_manuscript_has_no_known_unsupported_claims():
    manuscript = _manuscript_path()
    if not manuscript.exists():
        # The public reproducibility repository intentionally excludes
        # journal-submission source files. This guardrail remains active in
        # the private revision workspace where the manuscript is present.
        return
    text = manuscript.read_text(encoding="utf-8")
    for phrase in FORBIDDEN_UNSUPPORTED_CLAIMS:
        assert phrase not in text


def test_revised_manuscript_references_resolve():
    manuscript = _manuscript_path()
    if not manuscript.exists():
        return
    text = manuscript.read_text(encoding="utf-8")
    citations = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", text)
        for key in group.split(",")
    }
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    assert citations <= bibitems, sorted(citations - bibitems)
    references = set(re.findall(r"\\ref\{([^}]+)\}", text))
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    assert references <= labels, sorted(references - labels)
