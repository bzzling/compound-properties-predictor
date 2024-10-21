import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('data/data.csv')

# Step 2: Drop the "Minimum Degree" column
df = df.drop(columns=['Minimum Degree'])

# Step 3: Save the modified DataFrame back to the CSV file
df.to_csv('data/data.csv', index=False)