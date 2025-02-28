import pandas as pd

# Load the dataset
df = pd.read_csv('cwdata.csv')

# Remove rows where all elements are missing
df.dropna(how='all', inplace=True)

# Replace empty or whitespace strings with NaN to identify missing data
df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Drop rows with missing target values ('Fraud')
df.dropna(subset=['Fraud'], inplace=True)

# Optional: Convert target variable to a consistent format (binary)
df['Fraud'] = df['Fraud'].map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0})

# Identify and fill missing or invalid data with specific strategies
# Example: Fill missing 'Employed' values with 'N' assuming 'No' employment
df['Employed'].fillna('N', inplace=True)

# Example: Replace 'Home Owner' placeholder values ('1', '0') with 'Y', 'N'
df['Home Owner'] = df['Home Owner'].replace({'1': 'Y', '0': 'N'})

# Convert categorical data to consistent format (e.g., lowercase)
df['Area'] = df['Area'].str.lower()
df['Education'] = df['Education'].str.strip().str.lower()

# Handle missing numerical data, e.g., fill with median
for column in ['Income', 'Balance', 'Age']:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column].fillna(df[column].median(), inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Save cleaned data back to csv
df.to_csv('cwdata_cleaned.csv', index=False)


