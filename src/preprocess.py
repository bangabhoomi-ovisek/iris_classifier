import pandas as pd
from sklearn.model_selection import train_test_split

# Load the raw data we saved earlier
df = pd.read_csv('data/raw/iris.csv')

# X = the "questions" (measurements), y = the "answer" (species)
X = df.drop('species', axis=1)
y = df['species']

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save all 4 pieces into data/processed/
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Preprocessing done!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")