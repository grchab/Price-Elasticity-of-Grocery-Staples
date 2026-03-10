import os
import pandas as pd

# ============= Paths =============
MAIN_FOLDER = r"C:\Users\username\data\Economics"
income_file = os.path.join(MAIN_FOLDER, "Median_HH_Income_1995_2019.xlsx")

# ============= Load raw price data =============
bread_price = pd.read_csv(os.path.join(MAIN_FOLDER, "Average Price of Bread.csv"))
eggs_price  = pd.read_csv(os.path.join(MAIN_FOLDER, "Average Price of Eggs.csv"))
milk_price  = pd.read_csv(os.path.join(MAIN_FOLDER, "Average Price of Milk.csv"))

# ============= Load consumption data =============
loss_meat  = pd.read_csv(os.path.join(MAIN_FOLDER, "LossAdj - meat.csv"))
loss_dairy = pd.read_csv(os.path.join(MAIN_FOLDER, "LossAdj - Dairy.csv"))
loss_grain = pd.read_csv(os.path.join(MAIN_FOLDER, "LossAdj - grain.csv"))

# ============= Normalize price dates -> Year =============
for df in (bread_price, eggs_price, milk_price):
    df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
    df['Year'] = df['observation_date'].dt.year

# Yearly averages (column names are tolerated as given in your files)
bread_price_clean = (
    bread_price.groupby('Year', as_index=False)['Cost Per Pound']
    .mean().rename(columns={'Cost Per Pound': 'Avg_Bread_Price'})
)

eggs_price_clean = (
    eggs_price.groupby('Year', as_index=False)['Cost per dozen']
    .mean().rename(columns={'Cost per dozen': 'Avg_Egg_Price'})
)

milk_price_clean = (
    milk_price.groupby('Year', as_index=False)['Cost per gallon']
    .mean().rename(columns={'Cost per gallon': 'Avg_Milk_Price'})
)

# ============= Filter consumption series =============
eggs_cons = (
    loss_meat[
        (loss_meat['Commodity'].str.startswith('Eggs: Per capita availability adjusted for loss')) &
        (loss_meat['Attribute'] == 'Primary weight-Lbs/year')
    ][['Year', 'Value']]
    .rename(columns={'Value': 'Egg_Consumption_lb_per_capita'})
)

milk_cons = (
    loss_dairy[
        (loss_dairy['Commodity'] == 'All beverage milks: Per capita availability adjusted for loss') &
        (loss_dairy['Attribute'] == 'Primary weight-Lbs/year')
    ][['Year', 'Value']]
    .rename(columns={'Value': 'Milk_Consumption_lb_per_capita'})
)

bread_cons = (
    loss_grain[
        (loss_grain['Commodity'].str.startswith('Wheat flour: Per capita availability adjusted for loss')) &
        (loss_grain['Attribute'] == 'Primary weight-Lbs/year')
    ][['Year', 'Value']]
    .rename(columns={'Value': 'Bread_Consumption_lb_per_capita'})
)

# ============= Merge staples panels =============
merged = (
    bread_price_clean.merge(bread_cons, on='Year', how='inner')
    .merge(eggs_price_clean, on='Year', how='inner')
    .merge(eggs_cons,        on='Year', how='inner')
    .merge(milk_price_clean, on='Year', how='inner')
    .merge(milk_cons,        on='Year', how='inner')
)

# ============= Load & clean Household Income (Census) =============
# The sheet has repeated year blocks per race. We:
# 1) read everything after the header area,
# 2) extract a 4-digit Year from the first column,
# 3) choose the Median income column robustly,
# 4) keep the FIRST occurrence of each Year (ALL RACES block).

income_raw = pd.read_excel(income_file, header=None)  # no headers; treat all rows as data
income_raw = income_raw.iloc[6:, :].copy()            # skip top header rows (first valid years start at row 7)

# Extract 4-digit year from column 0; drop rows without a year
income_raw['Year'] = income_raw.iloc[:, 0].astype(str).str.extract(r'(\d{4})', expand=False)
income_raw = income_raw.dropna(subset=['Year'])
income_raw['Year'] = income_raw['Year'].astype(int)

# Try to locate Median income (estimate) column robustly.
# Typical positions: total ~15 columns; Median(Estimate) is often col index 11 (0-based) or -4.
median_col = None
candidate_idxs = [-4, 11, 12]  # try common positions first
for idx in candidate_idxs:
    if -income_raw.shape[1] <= idx < income_raw.shape[1]:
        s = pd.to_numeric(income_raw.iloc[:, idx], errors='coerce')
        if s.notna().sum() >= 30:
            median_col = idx
            break

# Fallback: pick the numeric column with most values in a sensible income range
if median_col is None:
    num = income_raw.apply(pd.to_numeric, errors='coerce')
    # Count entries that look like incomes (10k–200k)
    scores = num.apply(lambda s: s.between(10_000, 200_000).sum())
    median_col = scores.idxmax()

income_raw['Median_HH_Income'] = pd.to_numeric(income_raw.iloc[:, median_col], errors='coerce')

# Keep the FIRST occurrence of each year in sheet order -> corresponds to ALL RACES block
income_raw['year_first'] = income_raw.groupby('Year').cumcount()
income_df = income_raw.loc[income_raw['year_first'] == 0, ['Year', 'Median_HH_Income']].copy()

# Drop rows with no income value (rare but safe)
income_df = income_df.dropna(subset=['Median_HH_Income'])

# ============= Merge income into staples panel =============
merged = merged.merge(income_df, on='Year', how='inner')

# ============= Finalize & export =============
merged = merged.sort_values('Year').drop_duplicates(subset='Year')
output_path = os.path.join(MAIN_FOLDER, "Staples_Merged_With_Income.csv")
merged.to_csv(output_path, index=False)

print("Final merged file with household income saved to:")
print(output_path)

