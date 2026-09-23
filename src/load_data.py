import pandas as pd
from sklearn.datasets import load_iris

# Load the built-in Iris dataset
iris = load_iris()

# Convert it into a pandas DataFrame (like an Excel table)
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target  # 0, 1, 2 = the three flower types

# Save it as a CSV file into data/raw/
df.to_csv('data/raw/iris.csv', index=False)

print("Data saved successfully!")
print(df.head())  # shows first 5 rows to confirm it worked