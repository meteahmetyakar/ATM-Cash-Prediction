import pandas as pd
import holidays

indian_holidays = holidays.India()

df = pd.read_csv("ATM.csv")

df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True)

def get_day_type(date):
    if date in indian_holidays:
        holiday_name = indian_holidays.get(date)
        return holiday_name.replace(" (estimated)", "") if "(estimated)" in holiday_name else holiday_name
    return "Normal Day"

df["day_type"] = df["transaction_date"].apply(get_day_type)

print(df.head())

df.to_csv("ATM.csv", index=False)
