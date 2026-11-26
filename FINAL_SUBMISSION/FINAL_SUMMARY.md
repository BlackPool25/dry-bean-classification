# Final Submission Summary

Generated: 26 November 2026, 09:11

## Models implemented

| Model | Test accuracy | Test macro F1 | Weighted F1 | ROC-AUC (OvR) |
|---|---|---|---|---|
| RBF-SVM | 0.9258 | 0.9372 | 0.9258 | 0.9953 |
| Hist Gradient Boosting | 0.9229 | 0.9347 | 0.9229 | 0.9948 |
| Logistic Regression | 0.9214 | 0.9337 | 0.9217 | 0.9949 |
| Extra Trees | 0.9207 | 0.9328 | 0.9208 | 0.9943 |
| Random Forest | 0.9196 | 0.9314 | 0.9195 | 0.9936 |
| K-Nearest Neighbours | 0.9174 | 0.9300 | 0.9175 | 0.9870 |

## Selected model

- **Model:** RBF-SVM
- **Cross-validated macro F1:** 0.9444 +/- 0.0056 (5 stratified folds)
- **Held-out accuracy:** 0.9258
- **Held-out macro F1:** 0.9372
- **Held-out macro precision:** 0.9383
- **Held-out macro recall:** 0.9362
- **Held-out weighted F1:** 0.9258
- **ROC-AUC (one-vs-rest, macro):** 0.9953
- **Prediction time:** 167.4 ms for 2,723 beans
- **Hyperparameter search time:** 7.4 s
- **Hyperparameters:** `{"C": 100.0, "gamma": 0.01}`
- **Misclassified test beans:** 202 of 2723
- **Most frequent confusion:** SIRA -> DERMASON (51 beans)

## Artefacts

- LaTeX PDF: present, 56 pages
- Word report: present
- Saved pipeline: models/best_model.joblib
- Executed notebooks: 4
- Figures: 26
- Metric tables: 24

## Tests

- Command: `pytest -q`
- Result: **PASS** - ....................                                                     [100%]

## Unresolved limitations

- Single 80/20 split with a fixed seed; no repeated or nested CV.
- 68 duplicate rows retained (justified in Section 4.1).
- No external validation set from a second laboratory or camera.
- Five of seven varieties overlap; the residual error is geometric.
- LaTeX is compiled with Tectonic; the Word file requires Word or LibreOffice to refresh the table-of-contents field.
