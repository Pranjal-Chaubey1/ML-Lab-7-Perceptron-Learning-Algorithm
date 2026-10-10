# Seeds Classification using an Artificial Neural Network (ANN)

A machine learning notebook (ML Lab 6) that classifies seeds into 3 classes from 7 physical measurements, using a small **neural network built with TensorFlow / Keras**.

**Author:** Pranjal Chaubey
**Notebook:** `ML_Lab_6.ipynb`

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Key Insights](#key-insights)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

The goal is to predict the class (1, 2 or 3) of a seed from its measurements. The notebook covers:

- Data inspection (no cleaning was needed)
- Label shifting, stratified train-test split and feature scaling
- Building, compiling and training a feed-forward neural network with early stopping
- Plotting training and validation accuracy and loss
- Evaluating with accuracy, precision, recall, F1, classification report and confusion matrices
- A function that predicts the class of a new seed from user input

## Dataset

| Property | Value |
|---|---|
| File | `seeds_new.csv` |
| Rows | 210 |
| Columns | 8 (7 features + 1 target) |
| Target | `Class (1, 2, 3)` |
| Missing values | 0 |
| Duplicate rows | 0 |

**Features:** Area, Perimeter, Compactness, Length of kernel, Width of kernel, Asymmetry coefficient, Length of kernel groove.

All features are numeric (`float64`). The three classes look balanced: the stratified test set has exactly 14 samples per class, which points to about 70 per class overall (the notebook does not print the full class counts).

> The notebook reads the file from `/content/seeds_new (1) - seeds_new (1).csv` (a Google Colab path). Change the path in the `pd.read_csv(...)` cell to match where your CSV is stored.

## Tech Stack

- Python 3
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook / Google Colab

## Model Architecture

| Layer | Output shape | Parameters |
|---|---|---|
| Input | 7 | - |
| Dense (ReLU) | 16 | 128 |
| Dense (ReLU) | 8 | 136 |
| Dense (Softmax) | 3 | 27 |
| **Total** | | **291** |

**Training settings:**

| Setting | Value |
|---|---|
| Optimizer | Adam (learning rate 0.001) |
| Loss | `sparse_categorical_crossentropy` |
| Metric | accuracy |
| Epochs | up to 100 |
| Batch size | 16 |
| Validation split | 20% of the training data |
| Early stopping | monitor `val_loss`, patience 10, restore best weights |

## Project Workflow

1. **Load and inspect** the data (`head`, `tail`, `shape`, `info`, `describe`, `isnull`, `duplicated`).
2. **Separate** features `X` and target `y`.
3. **Shift labels** from 1, 2, 3 to 0, 1, 2 (needed for `sparse_categorical_crossentropy` and the softmax output).
4. **Split:** 80/20 with `random_state=42` and `stratify=y` (168 train / 42 test).
5. **Scale:** `StandardScaler` fitted on the training data only.
6. **Build** the Sequential network and **compile** it.
7. **Train** with `validation_split=0.2` and `EarlyStopping`.
8. **Plot** training vs validation accuracy and loss.
9. **Evaluate** on the test set and convert predictions back to the original classes (+1).
10. **Predict** the class of a new seed with `predict_new_seed()`.

## Results

**Training (epoch 100):** training accuracy 0.955, validation accuracy 0.941, training loss 0.080, validation loss 0.107. Early stopping never triggered, so all 100 epochs ran.

**Test set (42 samples):**

| Metric | Score |
|---|---|
| Test loss | 0.3827 |
| Accuracy | 0.857 |
| Precision (weighted) | 0.882 |
| Recall (weighted) | 0.857 |
| F1-score (weighted) | 0.845 |

**Per-class performance:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | 1.00 | 0.57 | 0.73 | 14 |
| 2 | 0.82 | 1.00 | 0.90 | 14 |
| 3 | 0.82 | 1.00 | 0.90 | 14 |

**Confusion matrix (rows = actual, columns = predicted):**

| | Pred 1 | Pred 2 | Pred 3 |
|---|---|---|---|
| **Actual 1** | 8 | 3 | 3 |
| **Actual 2** | 0 | 14 | 0 |
| **Actual 3** | 0 | 0 | 14 |

## Key Insights

- The dataset is clean and balanced, so almost no preparation was needed beyond scaling.
- Training and validation curves stay close together, so the network does not overfit.
- Classes 2 and 3 are classified perfectly; all 6 errors come from Class 1, which the model sometimes labels as Class 2 or 3.
- Test accuracy (0.857) is lower than validation accuracy (0.941), which is expected because both sets are very small (42 and 34 samples).
- The validation loss was still falling at epoch 100, so training for longer may help.

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter

# 2. Put the seeds CSV next to the notebook and update the path in the read_csv cell

# 3. Launch Jupyter and run all cells
jupyter notebook ML_Lab_6.ipynb
```

You can also upload the notebook and CSV to Google Colab and run it there. The last cell asks for 7 numbers through `input()`, so run it interactively.

## Project Structure

```
.
├── ML_Lab_6.ipynb     # Full analysis and model
├── seeds_new.csv      # Dataset (add this yourself)
└── README.md
```

## Limitations and Future Work

**Known issues in the current notebook**
- No random seed is set for TensorFlow / NumPy, so results change slightly on every run.
- The test set has only 42 samples, so one wrong prediction changes accuracy by about 2.4 points.
- `loss = "sparse_categorical_crossentropy"` is a leftover variable that the model does not use.
- `scaler.fit_transform(X_train)` is run a second time only to display the array (harmless, but unnecessary).
- Some imports (`LinearRegression`, `mean_squared_error`, `r2_score`, `Dropout`, `to_categorical`, `LabelEncoder`) are never used.

**Improvements to try**
- Set seeds with `tf.random.set_seed` and `np.random.seed` for repeatable results.
- Use k-fold cross-validation for a more reliable estimate.
- Tune the network (neurons, layers, `Dropout`, learning rate) and train for more epochs.
- Compare with Logistic Regression, KNN or Random Forest.
- Investigate which features make Class 1 overlap with Classes 2 and 3.
