# ML Assignment 1 — Polynomial Regression

**Student:** BT2024218

## Problems

| Problem | Description | Features | Max Degree | Regulariser | Best Degree |
|---------|-------------|----------|------------|-------------|-------------|
| var1 | Steam Turbine Optimisation | 6 | 10 | Lasso (L1) | 5 |
| var2 | Thermal Reservoir Mapping | 3 | 20 | Ridge (L2) | 10 |

## Results

| Problem | Hold-out MSE | Hold-out R² |
|---------|-------------|-------------|
| var1 | 0.3590 | 0.9609 |
| var2 | 0.2905 | 0.9917 |

## Repository Structure

```
part1.py                    # Training + inference for var1 (Lasso)
part2.py                    # Training + inference for var2 (Ridge)
report.pdf                  # Compiled report
BT2024218_pred_var1.csv     # Test predictions for var1
BT2024218_pred_var2.csv     # Test predictions for var2
part1_cv_curve.png          # CV MSE vs degree plot (var1)
part2_cv_curve.png          # CV MSE vs degree plot (var2)
BT2024218/                  # Datasets (train + test CSVs)
```

## How to Run

```bash
python part1.py   # generates BT2024218_pred_var1.csv
python part2.py   # generates BT2024218_pred_var2.csv
```

## Pipeline

1. 80/20 train/hold-out split
2. Degree sweep with 5-fold CV to tune regularisation
3. Best degree chosen by lowest CV MSE
4. Single evaluation on hold-out set
5. Refit on full training data → predict test set
