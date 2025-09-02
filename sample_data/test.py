import pandas as pd

# Load the dataset
df = pd.read_csv("Phishing_Email.csv")

# Print the first few rows and the column names
print("Columns:", df.columns.tolist())
print(df.head())
