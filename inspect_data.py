import pandas as pd

try:
# Google Sheet URL
    # format=csv exports the first sheet by default
    SHEET_URL = 'https://docs.google.com/spreadsheets/d/1vHnO2Zz3thN46-76j2fMMnp1W5efc9ottFLfwuOwwTY/export?format=csv'
    
    # Read directly from Google Sheets
    print(f"Reading data from: {SHEET_URL}")
    df = pd.read_csv(SHEET_URL)
    print("Columns:", df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
except Exception as e:
    print(f"Error reading file: {e}")
