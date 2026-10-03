# Late Delivery Prediction and Operational Segments

## Project Overview

This project is a practical Data Science and AI/ML notebook focused on predicting late deliveries and creating operational segments from the generated delivery dataset.

The notebook is organized into small, separate steps so that the calculations, preprocessing, machine learning models, clustering, and results are easy to follow and explain.

## Project Files
[▶️ Watch Project Demonstration Video]((https://drive.google.com/file/d/1XaAMPjt2uWeF4r3cyVUsMuGFpgVggM7r/view?usp=sharing)
)

- `campaign_response_project_exam_style (1).ipynb` — main project notebook covering data generation and loading, data auditing, descriptive statistics, a Welch t-test, confidence interval calculation, covariance/eigenvalue analysis, preprocessing and feature engineering, logistic regression, K-Means clustering, and an Artificial Neural Network (ANN).
- `data/raw/set_b.csv` — generated dataset created by the notebook when the data-generation cell is run.
- `outputs/` — folder used for saved summaries, predictions, clustering results, model comparisons, split information, figures, and package-version information.
- `models/` — folder used for saved preprocessing objects and the trained ANN model.

## Dataset

The notebook generates a synthetic delivery dataset with 300 original records and then adds 5 duplicate rows for the data-audit exercise.

The dataset contains the following main columns:

- `record_id` — unique record identifier
- `distance` — delivery distance
- `load` — delivery load
- `traffic` — traffic-related value
- `staff` — staff-related value
- `group` — operational group (`G1` or `G2`)
- `late` — target variable (`1` = late delivery, `0` = not late)

Missing values are intentionally introduced in the `distance` and `load` columns, and duplicate rows are included so that the notebook can demonstrate data auditing and cleaning.

The data is synthetic and is intended for practical/exam demonstration rather than real-world conclusions.

## Main Sections

### 1. Maths and Advanced Statistics

The notebook includes:

- Duplicate and missing-value checks
- Target-class counts
- Removal of exact duplicate records
- Fit, validation, and test data splitting
- Descriptive statistics for `distance`
- Histogram of distance values
- Welch independent-samples t-test comparing `distance` between `G1` and `G2`
- 95% confidence interval for the mean distance
- Covariance matrix calculation
- Eigenvalue and principal-direction analysis

### 2. Data Preprocessing and Feature Engineering

The preprocessing workflow includes:

- Separating predictors from the `late` target
- Median imputation for missing numeric values
- Creation of an engineered feature:

  `load / (staff + 1)`

- One-hot encoding of the `group` column
- Standard scaling of numeric features
- Saving the preprocessing objects for reuse

The preprocessing is fitted using the fit data and then applied to the validation and test data.

### 3. Supervised Learning

The project uses:

- A majority-class `DummyClassifier` as a baseline
- Logistic Regression as the main supervised model

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Test predictions and class-1 probabilities are also saved in the `outputs/` folder.

### 4. Unsupervised Learning

K-Means clustering is tested with:

- `k = 2`
- `k = 3`
- `k = 4`

The notebook records inertia and silhouette scores, selects the `k` value with the highest silhouette score, and creates a cluster profile using the numeric features.

Cluster labels are arbitrary identifiers and should not be interpreted as the `late` target classes.

### 5. Artificial Neural Network (ANN)

The notebook also contains an ANN for binary classification.

The network uses:

- Input layer based on the processed feature count
- Dense layer with 16 neurons and ReLU activation
- Dense layer with 8 neurons and ReLU activation
- Output layer with sigmoid activation

The model uses binary cross-entropy loss and the Adam optimizer.

Training includes validation data and early stopping. The ANN is evaluated on the same test records used for the other supervised models.

### 6. Saved Results

The notebook saves several results under `outputs/`, including:

- `splits.csv`
- `distance_summary.csv`
- `k_selection_scores.csv`
- `cluster_profiles.csv`
- `logistic_test_predictions.csv`
- `ann_test_predictions.csv`
- `model_comparison.csv`
- `library_versions.csv`

Figures are saved under `outputs/figures/`, and trained/preprocessing models are saved under `models/`.

## How to Run

### Windows

1. Make sure Python and Jupyter Notebook are installed.
2. Open the project folder in PowerShell or Command Prompt.
3. Start Jupyter Notebook:

   ```powershell
   jupyter notebook
   ```

4. Open `campaign_response_project_exam_style (1).ipynb`.
5. Run the notebook cells from top to bottom.

The notebook creates the required folders automatically when it starts.

## Important Notes

- The supplied data generator is kept unchanged.
- The dataset is synthetic.
- The notebook removes duplicate records before creating the model partitions.
- Fit, validation, and test records are kept separate.
- Missing numeric values are imputed using statistics learned from the fit data.
- Encoding and scaling are also fitted using the fit data.
- The test set is reserved for final model evaluation.
- Results from a single synthetic holdout dataset should not be treated as evidence of real-world performance.
- Before submission, run the notebook from a clean kernel and make sure all cells execute correctly.

## Project Purpose

The overall purpose of the project is to demonstrate a complete practical workflow covering data preparation, statistical analysis, feature engineering, supervised classification, unsupervised segmentation, and neural-network-based prediction for a late-delivery problem.
