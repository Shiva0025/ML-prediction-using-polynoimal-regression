import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROLL = "BT2024218"
TRAIN = f"{ROLL}/{ROLL}_train_var1.csv"
TEST  = f"{ROLL}/{ROLL}_test_var1.csv"
OUT   = f"{ROLL}_pred_var1.csv"
SEED  = 42
FEATS = ["x1", "x2", "x3", "x4", "x5", "x6"]

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
X, y  = train[FEATS].values, train["y"].values

X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, random_state=SEED)


def fit(X, y, deg):
    # polynomial regression
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Z    = poly.fit_transform(X)
    sc   = StandardScaler().fit(Z)
    Zs   = sc.transform(Z)
    alpha_max = np.max(np.abs(Zs.T @ (y - y.mean()))) / len(y)
    alphas    = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-3), 20)
    # lasso regularization
    model = LassoCV(
        alphas=alphas,
        cv=KFold(5, shuffle=True, random_state=SEED),
        tol=1e-3, max_iter=5000,
        random_state=SEED, n_jobs=-1,
    ).fit(Zs, y)
    return poly, sc, model


def predict(parts, X):
    poly, sc, model = parts
    return model.predict(sc.transform(poly.transform(X)))


print(f"{'Degree':>8} {'CV MSE':>14} {'Hold-out MSE':>14} {'Hold-out R²':>12}")
print("-" * 52)

rows = []
for d in range(1, 11):
    t0    = time.time()
    parts = fit(X_dev, y_dev, d)
    cv_mse = parts[2].mse_path_.mean(axis=1).min()

    p     = predict(parts, X_hold)
    h_mse = mean_squared_error(y_hold, p)
    h_r2  = r2_score(y_hold, p)

    rows.append(dict(degree=d, cv_mse=cv_mse, hold_mse=h_mse, hold_r2=h_r2))
    print(f"{d:>8} {cv_mse:>14.4f} {h_mse:>14.4f} {h_r2:>12.4f}   ({time.time()-t0:.0f}s)")

res  = pd.DataFrame(rows)
best = int(res.loc[res.cv_mse.idxmin(), "degree"])
print(f"\nBest degree: {best}  (CV MSE {res.cv_mse.min():.4f})")

final = fit(X, y, best)
pred  = predict(final, test[FEATS].values)
pd.DataFrame({"y": pred}).to_csv(OUT, index=False)
print(f"Saved → {OUT}  ({len(pred)} rows)")

res.to_csv("part1_cv_results.csv", index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3.5))
    plt.plot(res.degree, res.cv_mse, "o-")
    plt.axvline(best, ls="--", c="gray", label=f"best={best}")
    plt.xlabel("Polynomial degree"); plt.ylabel("CV MSE (log)")
    plt.yscale("log"); plt.legend(); plt.tight_layout()
    plt.savefig("part1_cv_curve.png", dpi=150)
except Exception as e:
    print("plot skipped:", e)