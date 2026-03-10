# grocery_staples_elasticity.py

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ===== Paths =====
MAIN_FOLDER = r"C:\Users\username\data\Economics"
MERGED_FILE = os.path.join(MAIN_FOLDER, "Staples_Merged_With_Income.csv")

# ===== Load merged wide dataset (Year-level, prices, quantities, income) =====
df_wide = pd.read_csv(MERGED_FILE)

# ===== Show income trend =====
plt.figure(figsize=(9, 4.5))
plt.plot(df_wide["Year"], df_wide["Median_HH_Income"], marker="o")
plt.title("Median Household Income (CPI-Adjusted) – All Races")
plt.xlabel("Year")
plt.ylabel("Dollars")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Reshape to long form: (Year, Food, Price, Quantity, Income) =====
spec = [
    ("Bread", "Avg_Bread_Price", "Bread_Consumption_lb_per_capita"),
    ("Eggs",  "Avg_Egg_Price",   "Egg_Consumption_lb_per_capita"),
    ("Milk",  "Avg_Milk_Price",  "Milk_Consumption_lb_per_capita"),
]

parts = []
for food, pcol, qcol in spec:
    sub = df_wide[["Year", pcol, qcol, "Median_HH_Income"]].copy()
    sub.columns = ["Year", "Price", "Quantity", "Median_HH_Income"]
    sub["Food"] = food
    parts.append(sub)

df = pd.concat(parts, ignore_index=True)

#Sort rows by Food and Year
df = df.sort_values(["Food", "Year"])

# ===== Run elasticity regressions and plot =====
foods = ["Bread", "Eggs", "Milk"]
fig, axes = plt.subplots(3, 4, figsize=(24, 12))  # 4th col = income panel
fig.suptitle("Elasticity Analysis with Household Income: Bread, Eggs, Milk", fontsize=18)

results = []

for i, food in enumerate(foods):
    sub = df[df["Food"] == food].copy()

    # Derived fields
    sub["Revenue"] = sub["Price"] * sub["Quantity"]
    sub["log_Q"] = np.log(sub["Quantity"])
    sub["log_P"] = np.log(sub["Price"])
    sub["log_Y"] = np.log(sub["Median_HH_Income"])

    X = sm.add_constant(sub[["log_P", "log_Y"]])
    y = sub["log_Q"]
    model = sm.OLS(y, X, missing="drop").fit()

    # Collect key metrics
    bP = model.params.get("log_P", np.nan)
    bY = model.params.get("log_Y", np.nan)
    ci = model.conf_int()
    ciP = ci.loc["log_P"].tolist() if "log_P" in ci.index else [np.nan, np.nan]
    ciY = ci.loc["log_Y"].tolist() if "log_Y" in ci.index else [np.nan, np.nan]
    results.append({
        "Food": food,
        "Obs": int(model.nobs),
        "R2": model.rsquared,
        "PriceElasticity": bP, "PE_CI_low": ciP[0], "PE_CI_high": ciP[1], "PE_p": model.pvalues.get("log_P", np.nan),
        "IncomeElasticity": bY, "IE_CI_low": ciY[0], "IE_CI_high": ciY[1], "IE_p": model.pvalues.get("log_Y", np.nan)
    })

    # Console summary
    print(f"\n\n========= {food} Elasticity Model =========")
    print(model.summary())

    #VISUALIZATIONS

    # ---- Col 1: Actual vs Predicted log(Q) ----
    sub["Pred_log_Q"] = model.predict(X)
    ax1 = axes[i, 0]
    ax1.plot(sub["Year"], sub["log_Q"], marker="o", label="Actual")
    ax1.plot(sub["Year"], sub["Pred_log_Q"], marker="x", label="Predicted")
    ax1.set_title(f"{food} - Actual vs Predicted log(Q)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("log(Q)")
    ax1.grid(True)
    ax1.legend()

    # ---- Col 2: Revenue trend ----
    ax2 = axes[i, 1]
    ax2.plot(sub["Year"], sub["Revenue"] / 1e6, marker="o")
    ax2.set_title(f"{food} - Revenue Trend")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Spending (Millions of $)")
    ax2.grid(True)

    # ---- Col 3: Demand curve (log Q vs log P) at mean income ----
    mean_income = sub["log_Y"].mean()
    price_range = np.linspace(sub["log_P"].min(), sub["log_P"].max(), 50)
    demand_curve = (
        model.params.get("const", 0.0)
        + model.params.get("log_Y", 0.0) * mean_income
        + model.params.get("log_P", 0.0) * price_range
    )
    ax3 = axes[i, 2]
    ax3.scatter(sub["log_P"], sub["log_Q"], alpha=0.6, label="Observed")
    ax3.plot(price_range, demand_curve, label="Fitted")
    ax3.set_title(f"{food} - Demand vs Price (log-log)")
    ax3.set_xlabel("log(Price)")
    ax3.set_ylabel("log(Quantity)")
    ax3.grid(True)
    ax3.legend()

    # ---- Col 4: Income response (log Q vs log Y) at mean price ----
    mean_price = sub["log_P"].mean()
    inc_grid = np.linspace(sub["log_Y"].min(), sub["log_Y"].max(), 50)
    inc_curve = (
        model.params.get("const", 0.0)
        + model.params.get("log_P", 0.0) * mean_price
        + model.params.get("log_Y", 0.0) * inc_grid
    )
    ax4 = axes[i, 3]
    ax4.scatter(sub["log_Y"], sub["log_Q"], alpha=0.6, label="Observed")
    ax4.plot(inc_grid, inc_curve, label="Fitted")
    ax4.set_title(f"{food} - Demand vs Income (log-log)")
    ax4.set_xlabel("log(Income)")
    ax4.set_ylabel("log(Quantity)")
    ax4.grid(True)
    ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ===== Elasticity summary table =====
df_results = pd.DataFrame(results)
print("\n\n=== Elasticity Summary (price & income) ===")
print(df_results)

out_res = os.path.join(MAIN_FOLDER, "Staples_Elasticity_Summary.csv")
df_results.to_csv(out_res, index=False)
print("Saved:", out_res)

