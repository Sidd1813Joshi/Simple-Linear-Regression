import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


FEATURE_COL = "Temperature (°C)"
TARGET_COL = "Sales ($)"

# Load data
df = pd.read_csv("data.csv")
X = df[[FEATURE_COL]]
y = df[TARGET_COL]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Evaluation
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Print results
print("\nModel trained successfully\n")
print(
    f"Model Equation: {TARGET_COL} = {model.intercept_:.2f} + {model.coef_[0]:.2f} × {FEATURE_COL}"
)
print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% variation explained)")
print(f"RMSE: {rmse:.2f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Plot
PLOT_TITLE = f"{FEATURE_COL} vs {TARGET_COL}"
X_LABEL = FEATURE_COL
Y_LABEL = TARGET_COL

plt.figure(figsize=(7, 5))
plt.scatter(df[FEATURE_COL], df[TARGET_COL])
plt.plot(df[FEATURE_COL], y_pred)

plt.title(PLOT_TITLE)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.tight_layout()

# Save plot
plt.savefig("scatter_plot.png", dpi=300)
print("Plot saved as: scatter_plot.png")
