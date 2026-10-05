# Credit Approval & Applicant Risk Analysis

## Project Objective
Analyze anonymized credit-card application data, identify approval/rejection patterns, and build a simple machine-learning classification model to predict approval status.

## Dataset
**UCI Credit Approval Dataset**

Source: https://archive.ics.uci.edu/dataset/27/credit+approval

The dataset contains 690 applications and 15 anonymized input attributes. The target attribute A16 represents the class: `+` (approved) or `-` (rejected).

## Key Findings from Initial Analysis
- Total applications: 690
- Approved applications: 307
- Rejected applications: 383
- Approval rate: 44.49%
- Missing values are represented by `?` in the original file and are handled during preprocessing.

## Methodology
1. Load the UCI dataset.
2. Handle missing values.
3. Separate categorical and numerical attributes.
4. Visualize approval outcomes.
5. Encode categorical variables and scale numerical variables.
6. Train a Logistic Regression classification model.
7. Evaluate using accuracy, precision, recall and F1-score.

## Project Files
- `project_code.py` — complete Python analysis and ML code
- `requirements.txt` — required Python libraries
- `README.md` — project documentation
- `Project_Report.docx` — project report

## Important Note
The UCI dataset uses anonymized feature names to protect confidentiality. Therefore, this project does not assign unsupported real-world meanings to A1-A15.

## Citation
Quinlan, J. (1987). Credit Approval [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5FS30
