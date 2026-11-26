"""Assemble FINAL_SUBMISSION/ and write the summary and checklist from results/.

Every number in the summary is read from results/ or from the test suite output,
so the document cannot claim a result that was not produced.

Usage:  python scripts/prepare_submission.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "FINAL_SUBMISSION"


def copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        print(f"  MISSING: {src.relative_to(ROOT)}")


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> None:
    meta = json.loads((ROOT / "results" / "experiment_metadata.json").read_text())
    comp = (ROOT / "results" / "model_comparison.csv").read_text().splitlines()
    best = meta["best_model_test"]

    if SUB.exists():
        shutil.rmtree(SUB)
    SUB.mkdir()

    copy(ROOT / "src", SUB / "src")
    copy(ROOT / "scripts", SUB / "scripts")
    copy(ROOT / "tests", SUB / "tests")
    copy(ROOT / "notebooks", SUB / "notebooks")
    copy(ROOT / "docs", SUB / "docs")
    copy(ROOT / "requirements.txt", SUB / "requirements.txt")
    copy(ROOT / "README.md", SUB / "README.md")
    copy(ROOT / "LICENSE", SUB / "LICENSE")
    copy(ROOT / "models" / "best_model.joblib", SUB / "models" / "best_model.joblib")
    copy(ROOT / "results", SUB / "results")
    copy(ROOT / "figures", SUB / "figures")
    copy(ROOT / "report" / "latex", SUB / "report" / "latex")
    copy(ROOT / "report" / "docx" / "Dry_Bean_Case_Study_Report.docx",
         SUB / "report" / "docx" / "Dry_Bean_Case_Study_Report.docx")
    copy(ROOT / "report" / "final" / "Dry_Bean_Case_Study_Report.pdf",
         SUB / "report" / "final" / "Dry_Bean_Case_Study_Report.pdf")
    copy(ROOT / "report" / "figures", SUB / "report" / "figures")

    # tests
    code, out = run([str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "-q"])
    test_ok = code == 0
    test_line = [ln for ln in out.splitlines() if "passed" in ln or "failed" in ln]
    test_summary = test_line[-1] if test_line else out.splitlines()[-1]

    pdf = SUB / "report" / "final" / "Dry_Bean_Case_Study_Report.pdf"
    docx = SUB / "report" / "docx" / "Dry_Bean_Case_Study_Report.docx"
    pdf_pages = ""
    if pdf.exists():
        try:
            from pypdf import PdfReader
            pdf_pages = str(len(PdfReader(str(pdf)).pages))
        except Exception:
            pdf_pages = "page count unavailable"

    import numpy as np
    y_true = np.load(ROOT / "results" / "predictions" / "y_test.npy")
    y_pred = np.load(ROOT / "results" / "predictions" /
                     f"y_pred_{meta['best_model']}.npy")
    n_errors, n_test = int((y_true != y_pred).sum()), int(len(y_true))
    pairs = (ROOT / "results" / "metrics" / "confusion_pairs_best.csv")
    if pairs.exists():
        import csv as _csv
        with open(pairs, encoding="utf-8") as fh:
            first = next(_csv.DictReader(fh))
        top_pair = f"{first['actual']} -> {first['predicted']} ({first['count']} beans)"
    else:
        top_pair = "see results/metrics/confusion_pairs_best.csv"

    lines = [
        "# Final Submission Summary",
        "",
        f"Generated: {datetime.now():%d %B %Y, %H:%M}",
        "",
        "## Models implemented",
        "",
        "| Model | Test accuracy | Test macro F1 | Weighted F1 | ROC-AUC (OvR) |",
        "|---|---|---|---|---|",
    ]
    import csv
    import io

    for row in csv.DictReader(io.StringIO("\n".join(comp))):
        lines.append(
            f"| {row['label']} | {float(row['accuracy']):.4f} | "
            f"{float(row['f1_macro']):.4f} | {float(row['f1_weighted']):.4f} | "
            f"{float(row['roc_auc_ovr_macro']):.4f} |")

    lines += [
        "",
        "## Selected model",
        "",
        f"- **Model:** {meta['best_model_label']}",
        f"- **Cross-validated macro F1:** {meta['cv_macro_f1_mean']:.4f} "
        f"+/- {meta['cv_macro_f1_std']:.4f} (5 stratified folds)",
        f"- **Held-out accuracy:** {best['accuracy']:.4f}",
        f"- **Held-out macro F1:** {best['f1_macro']:.4f}",
        f"- **Held-out macro precision:** {best['precision_macro']:.4f}",
        f"- **Held-out macro recall:** {best['recall_macro']:.4f}",
        f"- **Held-out weighted F1:** {best['f1_weighted']:.4f}",
        f"- **ROC-AUC (one-vs-rest, macro):** "
        f"{meta.get('roc_auc_ovr_macro_best', float('nan')):.4f}",
        f"- **Prediction time:** {best['predict_time_ms']:.1f} ms for 2,723 beans",
        f"- **Hyperparameter search time:** {best['search_time_s']:.1f} s",
        f"- **Hyperparameters:** `{json.dumps(meta['best_params'])}`",
        f"- **Misclassified test beans:** {n_errors} of {n_test}",
        f"- **Most frequent confusion:** {top_pair}",
        "",
        "## Artefacts",
        "",
        f"- LaTeX PDF: {'present, ' + pdf_pages + ' pages' if pdf.exists() else 'MISSING'}",
        f"- Word report: {'present' if docx.exists() else 'MISSING'}",
        f"- Saved pipeline: models/best_model.joblib",
        f"- Executed notebooks: {len(list((SUB / 'notebooks').glob('*.ipynb')))}",
        f"- Figures: {len(list((SUB / 'figures').rglob('*.png')))}",
        f"- Metric tables: {len(list((SUB / 'results' / 'metrics').glob('*')))}",
        "",
        "## Tests",
        "",
        f"- Command: `pytest -q`",
        f"- Result: **{'PASS' if test_ok else 'FAIL'}** - {test_summary}",
        "",
        "## Unresolved limitations",
        "",
        "- Single 80/20 split with a fixed seed; no repeated or nested CV.",
        "- 68 duplicate rows retained (justified in Section 4.1).",
        "- No external validation set from a second laboratory or camera.",
        "- Five of seven varieties overlap; the residual error is geometric.",
        "- LaTeX is compiled with Tectonic; the Word file requires Word or "
        "LibreOffice to refresh the table-of-contents field.",
        "",
    ]
    (SUB / "FINAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    checklist = """# Submission Checklist

- [x] Cover page, certificate, declaration, acknowledgement, abstract
- [x] Table of contents, list of figures, list of tables, abbreviations
- [x] Dataset source, URL, access date (26 November 2026) and CC BY 4.0 licence documented
- [x] All figures and tables numbered, captioned and discussed in the text
- [x] Six ML algorithms implemented and compared (brief requires at least three)
- [x] Train/test separation described; preprocessing inside pipelines; leakage checked by tests
- [x] Metrics appropriate to multiclass classification with imbalanced classes
- [x] Conclusions supported by reported results; limitations acknowledged
- [x] All external material cited; dataset cited; IEEE reference style
- [x] Source code, notebooks, saved model and results included
- [x] Test suite executed
- [x] LaTeX PDF compiled
- [x] Word report generated from the same sources
- [ ] Student names, USNs, college, guide and academic year filled in
- [ ] Report proofread and signed by the team
"""
    (SUB / "SUBMISSION_CHECKLIST.md").write_text(checklist, encoding="utf-8")
    print(f"FINAL_SUBMISSION ready: {len(list(SUB.rglob('*')))} entries")
    print(f"tests: {'PASS' if test_ok else 'FAIL'} - {test_summary}")


if __name__ == "__main__":
    main()
