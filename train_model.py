import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# Load the Wheat Seeds dataset
column_names = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry', 'Groove', 'Class']
df = pd.read_csv("Data/seeds_dataset.txt", delim_whitespace=True, header=None, names=column_names)

# Check for any malformed rows and drop them
df = df.dropna()  # Drop rows with missing or malformed data

# Split the data into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save results to a JSON file
results = {
        "accuracy": accuracy,
            "feature_importances": list(model.feature_importances_)
}
with open("results.json", "w") as f:
        json.dump(results, f)

print("Model training completed. Results saved to results.json.")