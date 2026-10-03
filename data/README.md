# Telecom churn data

The report cites **mnassrib, Telecom Churn Dataset (Kaggle)**, accessed December 19, 2025. The supplied snapshots are retained without changing values or splitting observations again:

- `raw/train.csv`: 2,666 rows, 20 columns; 388 churners.
- `raw/test.csv`: 667 rows, 20 columns; 95 churners.
- `processed/cleaned_train.csv` and `cleaned_test.csv`: the saved encoded/scaled splits, with 15 predictors plus `Churn`.

The report identifies the source by title and author; search Kaggle for that exact dataset to obtain a new copy. The local snapshots are sufficient for this analysis. No original dataset license file was supplied; attribution here does not grant new redistribution rights.

The included schema has state, area code, account length, plan flags, usage, support-call counts, and churn. It has no customer names, telephone numbers, email addresses, or customer IDs. The snapshots are small (under 1 MB each). The processed splits are intentionally retained because they are the exact historical inputs to the reported model comparison.
