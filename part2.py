import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROLL  = "BT2024218"
TRAIN = f"{ROLL}/{ROLL}_train_var2.csv"
TEST  = f"{ROLL}/{ROLL}_test_var2.csv"
OUT   = f"{ROLL}_pred_var2.csv"
SEED  = 42
FEATS = ["x1", "x2", "x3"]
ALPHAS = np.logspace(-6, 4, 40)

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
X, y  = train[FEATS].values, train["y"].values

X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, random_state=SEED)


def fit(X, y, deg):
    # polynomial regression
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Zs   = StandardScaler().fit_transform(poly.fit_transform(X))
    # ridge regularization
    model = RidgeCV(
        alphas=ALPHAS,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="neg_mean_squared_error",
    ).fit(Zs, y)
    return poly, StandardScaler().fit(poly.transform(X)), model


def fit_full(X, y, deg):
    # polynomial regression
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Z    = poly.fit_transform(X)
    sc   = StandardScaler().fit(Z)
    Zs   = sc.transform(Z)
    # ridge regularization
    model = RidgeCV(
        alphas=ALPHAS,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="neg_mean_squared_error",
    ).fit(Zs, y)
    return poly, sc, model


def predict(parts, X):
    poly, sc, model = parts
    return model.predict(sc.transform(poly.transform(X)))


print(f"{'Degree':>8} {'CV MSE':>14} {'Hold-out MSE':>14} {'Hold-out R²':>12}")
print("-" * 52)

rows = []
for d in range(1, 21):
    t0    = time.time()
    # polynomial regression + ridge per degree
    poly  = PolynomialFeatures(degree=d, include_bias=False)
    Z_dev = poly.fit_transform(X_dev)
    sc    = StandardScaler().fit(Z_dev)
    Zs    = sc.transform(Z_dev)
    # ridge regularization
    model = RidgeCV(
        alphas=ALPHAS,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="neg_mean_squared_error",
    ).fit(Zs, y_dev)

    cv_mse = -model.best_score_
    p      = model.predict(sc.transform(poly.transform(X_hold)))
    h_mse  = mean_squared_error(y_hold, p)
    h_r2   = r2_score(y_hold, p)

    rows.append(dict(degree=d, cv_mse=cv_mse, hold_mse=h_mse, hold_r2=h_r2))
    print(f"{d:>8} {cv_mse:>14.4f} {h_mse:>14.4f} {h_r2:>12.4f}   ({time.time()-t0:.0f}s)")

res  = pd.DataFrame(rows)
best = int(res.loc[res.cv_mse.idxmin(), "degree"])
print(f"\nBest degree: {best}  (CV MSE {res.cv_mse.min():.4f})")

final = fit_full(X, y, best)
pred  = predict(final, test[FEATS].values)
pd.DataFrame({"y": pred}).to_csv(OUT, index=False)
print(f"Saved → {OUT}  ({len(pred)} rows)")

res.to_csv("part2_cv_results.csv", index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3.5))
    plt.plot(res.degree, res.cv_mse, "o-")
    plt.axvline(best, ls="--", c="gray", label=f"best={best}")
    plt.xlabel("Polynomial degree"); plt.ylabel("CV MSE (log)")
    plt.yscale("log"); plt.legend(); plt.tight_layout()
    plt.savefig("part2_cv_curve.png", dpi=150)
except Exception as e:
    print("plot skipped:", e)