import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

print("Starting Loan Model Training...")

# Load dataset
df = pd.read_csv("train.csv")

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# Print columns to check
print("Columns in dataset:")
print(df.columns)

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Find correct target column automatically
target_column = None
for col in df.columns:
    if "loan" in col.lower() and "status" in col.lower():
        target_column = col
        break

if target_column is None:
    print("Loan Status column not found. Check column names above.")
    exit()

print("Target column detected:", target_column)

# Convert target
df[target_column] = df[target_column].map({"Y": 1, "N": 0})

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("loan_model.pkl", "wb"))

print("Model saved successfully!")
