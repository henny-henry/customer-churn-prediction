# Research provenance and limitations

The report's Table 2 matches the historical evaluation CSV and notebook outputs at the displayed precision. All seven models and the original full-search grids are preserved. The README figures were extracted directly from notebook PNG outputs, not captured from PDF pages. The complete CSV includes serialized arrays as text; it is an archival result export, not a model checkpoint, and should never be loaded with `eval`.

## Methodology caveats

- The preprocessing step fits the encoder/scaler on the entire training split before cross-validation. The test data is transformed with training statistics, but each CV validation fold contributes to the shared preprocessing fit. A fold-local pipeline would address this; applying one would be a new experiment and could change results.
- State is ordinal encoded, area code is treated numerically, and all numeric predictors (including encoded categories) are standardized. No one-hot conversion was introduced.
- The report describes broader search dimensions than the code implements. Actual grids cover tree depth/minimum split size, forest depth/tree count, SVM C/kernel, KNN neighbor count, ridge alpha, and bagging estimator count. They do not tune tree minimum leaf size, forest max-features, SVM gamma, KNN metric/weights, or bagging base learner.
- The paper's summary-statistics table uses original-scale features; the saved notebook's summary cell comes after scaling. It is not a direct regeneration of that table.
- PCA is not a modeling input. Importance bars for linear models are absolute coefficients, not causal effects. There are no confidence intervals supporting a meaningful difference between the top two AUCs.

## Engineering changes

Fixed filename case for cross-platform use, made notebook paths root-relative, moved shared plotting functions into `src/evaluation.py`, removed unused imports/interactive reloads, added support for pandas string dtypes, and corrected a data-overview helper that referred to a global variable. Caches and irrelevant warning output were removed. Original numeric outputs, train/test rows, target values, full search settings, and historical CSVs remain intact.

Default smoke mode reduces only validation workload. It uses small stratified subsets and two-fold searches with one original candidate per grid; it is not an additional research result. Full experiment execution requires explicitly turning smoke mode off. The large exploratory pairplot is skipped in smoke mode. Fresh exports go to an ignored rerun directory.

## Validation

Both notebooks pass nbformat validation and sequential execution in smoke mode on Python 3.12. Preprocessing regenerates both saved processed splits within 1e-12 tolerance. All seven estimators, metric helpers, searches, and export paths were exercised on the reduced samples; the full 10-fold experiments were not rerun. Tested with NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, Matplotlib 3.11.2, and seaborn 0.13.2. SVC probability estimation is deprecated in that version; the dependency upper bound preserves the original API. Historical scores remain unchanged.
