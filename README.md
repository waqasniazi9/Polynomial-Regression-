# AdVision AI — Capstone-03
> Polynomial Regression Marketing Mix Modeling (MMM) & Advertising Sales Predictor with an Executive Streamlit Web App.

---

## 🎯 Project Overview
This project builds a **Polynomial Regression model** to predict product sales based on advertising budgets across **TV**, **Radio**, and **Newspaper**, specifically capturing non-linear relationships, channel synergy interactions, and diminishing marginal returns at scale.

### Key Value & Business Problem Solved:
1. **Optimize Advertising Budgets for Maximum ROI**: Uses mathematical optimization (`scipy.optimize.minimize` SLSQP) to find the global revenue-maximizing budget distribution across channels for any target spend.
2. **Identify Overspending & Diminishing Returns**: Visualizes continuous first derivatives ($\frac{\Delta\text{Sales}}{\Delta\text{Spend}}$) and curvature ($\frac{d^2\text{Sales}}{d\text{Spend}^2}$), highlighting the saturation point on TV where additional spending produces flattening returns.
3. **Allocate Budgets Across Multiple Channels**: Captures cross-channel marketing synergy ($TV \times Radio$ interaction term = $+1.39$), proving that running coordinated TV and radio campaigns yields super-linear sales lift.
4. **Data-Driven Decision Making**: Instant real-time simulation with interactive sliders, strategic presets, sensitivity matrices, and downloadable CSV prediction plans.
5. **Understand Non-Linearity in Real-World Sales**: Directly compares Linear Regression ($R^2 = 90.59\%$) against Polynomial Regression ($R^2 = 95.33\%$), highlighting why polynomials are essential for modeling advertising saturation.

---

## 🚀 Setup & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12 recommended.

### 1. Virtual Environment & Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
*(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### 2. Train Model & Save Artifacts
```powershell
python train_model.py
```
Outputs model metrics, cross-validation selection, and generates artifacts in `artifacts/`:
- `polynomial_pipeline.joblib`: Serialized Degree-2 pipeline (PolynomialFeatures + StandardScaler + LinearRegression).
- `metrics.json`: Train/Test MAE, RMSE, and $R^2$ evaluations.
- `degree_selection.csv`: 5-fold cross-validation RMSE comparing degrees 1, 2, and 3.
- `test_predictions.csv`: Actual vs predicted test predictions with residuals.

### 3. Launch Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dataset Information
- **Source**: [Kaggle Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset)
- **Observations**: 200 campaign records.
- **Features**:
  - `TV`: Advertising spend on TV (dataset units)
  - `Radio`: Advertising spend on Radio (dataset units)
  - `Newspaper`: Advertising spend on Newspaper (dataset units)
- **Target**:
  - `Sales`: Units of product sold

---

## 🔬 Modeling Methodology & Evaluation
- **Split**: 80% Train (160 rows), 20% Test (40 rows), `random_state=42`.
- **Model Selection**: 5-Fold Shuffled Cross-Validation on training data only:
  - Degree 1 (Linear): CV RMSE = 1.67
  - **Degree 2 (Quadratic)**: **CV RMSE = 1.34** (Selected — lowest error, guards against overfitting)
  - Degree 3 (Cubic): CV RMSE = 1.63 (Overfitting / boundary explosion)
- **Holdout Test Set Performance**:
  - **Polynomial (Degree 2)**: **$R^2 = 0.9533$**, **MAE = $0.903$**, **RMSE = $1.201$**
  - **Linear Baseline (Degree 1)**: $R^2 = 0.9059$, MAE = $1.275$, RMSE = $1.705$
  - **Improvement**: ~29% reduction in Mean Absolute Error over standard linear regression.

---

## 📉 Diminishing Returns & Mathematical Insights
The fitted polynomial equation features provide direct empirical marketing insights:
- **$TV$ (+6.41)**: Primary linear sales driver.
- **$TV^2$ (-2.69)**: Negative quadratic coefficient confirms concave diminishing marginal returns. Pouring excessive budget into TV without diversification leads to saturation.
- **$TV \times Radio$ (+1.39)**: Positive interaction coefficient proves that pairing TV with Radio generates compounding marketing synergy.
- **$Newspaper$ (-0.04)**: Minimal marginal contribution; allocating excessive budget to print yields low return.

---

## 🖥️ Streamlit Web Application Features
1. **Executive KPI Dashboard**: Instant sales forecast, non-linear advantage over linear baseline, efficiency ratio, and TV saturation status.
2. **Sidebar Controls & Presets**: Live sliders with historical boundary guardians and one-click strategy presets (*Median Plan, Aggressive Growth, TV-Heavy, Radio Pulse, Balanced Split*).
3. **Tab 1: Instant Prediction & Scenario Lab**: Live plan table, CSV export, and What-If sensitivity tests (+10% boosts, budget reallocation).
4. **Tab 2: Diminishing Returns & Saturation Analytics**: Interactive response curves, marginal sales first derivatives, saturation alerts, and an interactive **3D TV × Radio synergy response surface**.
5. **Tab 3: AI Budget Allocator & SLSQP Optimizer**: User sets total budget target; algorithm calculates the optimal channel split and quantifies the expected sales lift (+15% to +25% lift for the same dollar spend).
6. **Tab 4: Model Evaluation & Diagnostics**: 5-fold CV comparisons, actual vs predicted scatter with 45° line, residual analysis, and feature weight breakdowns.
7. **Tab 5: Dataset & Market Analysis**: Interactive data table, summary statistics, and correlation heatmaps.

---

## 📁 Repository Structure
```
├── advertising.csv               # Kaggle advertising dataset
├── data/
│   └── advertising.csv           # Redundant copy for standardized loading
├── train_model.py                # Model training, CV evaluation, optimizer & helpers
├── app.py                        # Modern Streamlit application
├── requirements.txt              # Project dependencies
├── artifacts/                    # Saved pipeline, metrics.json, cv_results.csv
├── README.md                     # Comprehensive documentation
└── LINKEDIN_POST.txt             # Ready-to-use portfolio post template
```
