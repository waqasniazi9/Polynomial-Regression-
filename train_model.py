from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
FEATURES = ["TV", "Radio", "Newspaper"]
def pipeline(degree):
    return Pipeline([("polynomial", PolynomialFeatures(degree=degree, include_bias=False)), ("scale", StandardScaler()), ("regression", LinearRegression())])
def load_data():
    primary = Path(__file__).parent / "data" / "advertising.csv"
    if primary.exists():
        return pd.read_csv(primary)
    fallback = Path(__file__).parent / "advertising.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError("Could not find advertising.csv in 'data/' or current directory.")
def train():
    data = load_data()
    if data[FEATURES + ["Sales"]].isna().any().any():
        raise ValueError("Missing values require explicit handling.")
    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data.Sales, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for degree in [1,2,3]:
        mse = -cross_val_score(pipeline(degree), X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        rows.append({"Degree": degree, "CV_RMSE": float(np.sqrt(mse).mean()), "CV_RMSE_SD": float(np.sqrt(mse).std())})
    cv_results = pd.DataFrame(rows)
    degree = int(cv_results.loc[cv_results.Degree > 1].sort_values("CV_RMSE").iloc[0].Degree)
    model = pipeline(degree).fit(X_train, y_train)
    linear = pipeline(1).fit(X_train, y_train)
    def evaluate(estimator):
        p = estimator.predict(X_test)
        return {"MAE": float(mean_absolute_error(y_test,p)), "RMSE": float(np.sqrt(mean_squared_error(y_test,p))), "R2": float(r2_score(y_test,p))}
    report = {"selected_degree": degree, "train_rows": len(X_train), "test_rows": len(X_test), "polynomial": evaluate(model), "linear": evaluate(linear)}
    comparison = X_test.copy()
    comparison["Actual"] = y_test
    comparison["Polynomial"] = model.predict(X_test)
    comparison["Linear"] = linear.predict(X_test)
    comparison["Residual"] = comparison.Actual - comparison.Polynomial
    return model, linear, report, cv_results, comparison, data, X_train

def response(model, training, budgets, channel):
    grid = np.linspace(training[channel].min(), training[channel].max(), 150)
    frame = pd.DataFrame({col: np.repeat(budgets[col], len(grid)) for col in FEATURES})
    frame[channel] = grid
    prediction = model.predict(frame)
    marginal = np.gradient(prediction, grid)
    curvature = np.gradient(marginal, grid)
    return pd.DataFrame({"Budget": grid, "Predicted sales": prediction, "Marginal sales": marginal, "Curvature": curvature})

def optimize_budget(total_budget, model, training_data, min_per_channel=0.0):
    from scipy.optimize import minimize
    bounds = [(min_per_channel, float(training_data[ch].max())) for ch in FEATURES]
    x0 = [min(max(total_budget / len(FEATURES), b[0]), b[1]) for b in bounds]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    
    def objective(x):
        df = pd.DataFrame([x], columns=FEATURES)
        return -float(model.predict(df)[0])
        
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints)
    
    alloc = {ch: max(0.0, round(float(val), 2)) for ch, val in zip(FEATURES, res.x)}
    pred = float(model.predict(pd.DataFrame([[alloc[ch] for ch in FEATURES]], columns=FEATURES))[0])
    return alloc, pred

def get_feature_breakdown(model):
    poly = model.named_steps['polynomial']
    reg = model.named_steps['regression']
    feature_names = poly.get_feature_names_out(FEATURES)
    coefs = reg.coef_
    
    types = []
    for f in feature_names:
        if "^2" in f:
            types.append("Non-Linear Curvature")
        elif " " in f:
            types.append("Synergy Interaction")
        else:
            types.append("Direct Linear Effect")
            
    df = pd.DataFrame({
        "Feature": feature_names,
        "Type": types,
        "Coefficient": coefs,
        "Impact": [abs(c) for c in coefs]
    }).sort_values("Impact", ascending=False).reset_index(drop=True)
    return df

def generate_synergy_grid(model, training_data, newspaper_val=0.0, resolution=30):
    tv_vals = np.linspace(training_data["TV"].min(), training_data["TV"].max(), resolution)
    radio_vals = np.linspace(training_data["Radio"].min(), training_data["Radio"].max(), resolution)
    TV_grid, Radio_grid = np.meshgrid(tv_vals, radio_vals)
    
    flat_tv = TV_grid.ravel()
    flat_radio = Radio_grid.ravel()
    flat_news = np.full_like(flat_tv, newspaper_val)
    
    df_grid = pd.DataFrame({"TV": flat_tv, "Radio": flat_radio, "Newspaper": flat_news})
    sales_pred = model.predict(df_grid).reshape(resolution, resolution)
    return tv_vals, radio_vals, sales_pred

if __name__ == "__main__":
    import json, joblib
    model, linear, report, cv_results, comparison, data, training = train()
    folder = Path(__file__).parent / "artifacts"
    folder.mkdir(exist_ok=True)
    joblib.dump(model, folder / "polynomial_pipeline.joblib")
    (folder / "metrics.json").write_text(json.dumps(report, indent=2))
    cv_results.to_csv(folder / "degree_selection.csv", index=False)
    comparison.to_csv(folder / "test_predictions.csv", index=False)
    print(json.dumps(report, indent=2))
