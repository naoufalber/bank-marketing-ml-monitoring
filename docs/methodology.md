# Evaluation and limitations

## Current executable benchmark

`src/train.py` uses a fixed 80/20 stratified split (seed 42). Imputation, scaling and one-hot encoding are fitted inside each model pipeline on the training partition. `duration` is removed by default. Both model specifications are fixed before holdout evaluation. F1 is measured at the default probability threshold of 0.5; it is not a tuned business threshold.

The dataset is date-ordered, but this benchmark uses a random stratified split for comparability with the academic exercise. A deployment study should add chronological validation, calibration, cost-based decision thresholds and a feature-availability audit. No production performance claim is made.

The synthetic drift output changes `euribor3m` and `cons.conf.idx`, while keeping `age` as a control. PSI uses unique quantile edges, infinite tails and normalized smoothing. These are input-distribution diagnostics; they do not establish how real subscription outcomes would change under an economic shock.

## Historical notebook

The notebook retains earlier outputs for learning curves, 5-fold cross-validation, bootstrap uncertainty, randomized search, permutation importance, SHAP and drift mitigation. Its reported CV ROC-AUC of 0.9448 includes post-call `duration`; it must not be compared as a pre-call result to the current CLI.

The research PSI implementation does not cover all shifted-tail mass. Its mitigation comparison reuses a fitted preprocessing object and evaluates strategies on different populations. Those historical values are retained as academic records, not validation of the new implementation. The research notebook should be rerun and revised before using its mitigation ranking for a decision.

## Validation performed

The current CLI completed on all 41,188 supplied observations, using 32,950 training and 8,238 held-out records. Outputs are in `reports/validated/`. The original SHAP notebook was inspected but not rerun end to end. Dependency versions for the CLI are pinned in `requirements.txt`; the research extras are unpinned and may require compatibility adjustment.
