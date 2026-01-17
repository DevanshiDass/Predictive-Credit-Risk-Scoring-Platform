import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(url)

print("Rows, Columns:", df.shape)
print(df.head())

# Print columns to identify target
print("\nColumn names:\n", list(df.columns))
