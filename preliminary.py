import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ATM.csv")

print("Dataset Informations :\n")
print(df.info())
print("\nFirst Rows: \n")
print(df.head())
print("\nStatistics: \n")
print(df.describe())

daily_withdrawals = df.groupby("transaction_date")["total_amount_withdrawn"].sum()
daily_withdrawals.plot(title="Daily Withdrawn Size", figsize=(10, 5), color="blue")
plt.xlabel("Date")
plt.ylabel("Total withdrawn")
plt.grid()
plt.show()

weekly_withdrawals = df.groupby("weekday")["total_amount_withdrawn"].mean()
weekly_withdrawals.sort_index().plot(kind="Bar", ", figsize=(8, 4), color="orange")
plt.xlabel("Weekdays")
plt.ylabel("Average withdrawn size")
plt.grid(axis='y')
plt.show()

missing_values = df.isnull().sum()
print("\n Values:\n")
print(missing_values)

transactions_vs_amount = df.groupby("transaction_date").agg({
    "No_Of_Withdrawals": "sum",
    "total_amount_withdrawn": "sum"
})
print("\nDaily Withdrawn Count and Withdrawn Amount:\n")
print(transactions_vs_amount.head())

df = pd.get_dummies(df, columns=["weekday", "working_day", "day_type"], drop_first=True)

df["moving_avg_7d"] = df["total_amount_withdrawn"].rolling(window=7).mean()

df["prev_day_withdrawals"] = df["No_Of_Withdrawals"].shift(1)
df["prev_day_amount"] = df["total_amount_withdrawn"].shift(1)

df["lag_1"] = df["total_amount_withdrawn"].shift(1)
df["lag_7"] = df["total_amount_withdrawn"].shift(7)

df.to_csv("updated_ATM.csv", index=False)
