# Challenge Lab 3: Automate ML Workflows: CI/CD for Wheat Seeds Model with GitHub Actions

### Lab overview

In this lab, you will:

- Set up a CI/CD pipeline for the wheat seeds dataset using GitHub Actions
- Train a machine learning model using scikit-learn
- Generate and deploy the model results as an HTML page to GitHub pages

Estimated completion time

45 minutes

---

## Task 1: Setting up the project repository and loading the dataset

This task aims to create a GitHub repository and set up this lab’s basic project folder structure.

1. Open your Web browser, log in to GitHub (if necessary), and create a new repository, e.g., wheat-seeds-html-ci-cd add a description, and initialize with a README.
2. Open the Visual Studio Code, open a new terminal window and clone the repository locally, then switch to the folder.
3. Open the project folder in VS Code and launch a new terminal.
4. Now, create a new virtual environment for this task and activate it. Run the following command in the terminal window to create the virtual environment.
5. Once the new virtual environment is set up, install the required libraries in the venv as follows.

```bash
pip install scikit-learn pandas matplotlib
```

6. Create a necessary folder structure as below.

```
Wheat-seeds-html-ci-cd/
├── Data/
└── seeds_dataset.txt
├── train_model.py
├── generate_html.py
├── requirements.txt
├── .github/
└── workflows/
└── ci.yml
├── index.html

# Will be auto-generated

├── README.md
```

7. Execute the command below in the terminal window to download the dataset in the data folder. A copy has been provided in C:\MLOps\Data-Files, if download is an issue.

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt -P Data/
```

---

## Task 2: Training a model and generating an HTML app

This task aims to process the dataset, train a model, and write a script to automatically generate the HTML file in the project folder which will be used later to show the results.

1. Create a new file with the name `train_model.py` in the project folder and type in the following code.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# Load the Wheat Seeds dataset
column_names = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width',
'Asymmetry', 'Groove', 'Class']
df = pd.read_csv("Data/seeds_dataset.txt", delim_whitespace=True,
header=None, names=column_names)

# Check for any malformed rows and drop them
df = df.dropna()

# Drop rows with missing or malformed data

# Split the data into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)

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
```

This script will train the RandomForest model and save the results (accuracy, feature importances) to a JSON file for later use.

2. Run the `train_model.py` python script in the VS Code terminal.
3. Verify that `results.json` is generated and contains model accuracy and feature importances after executing the above model training script.

4. Create a new Python file in the project folder with the name `generate_html.py`. This script will read the `results.json` file and generate an HTML file (`index.html`) for the app.

```python
import json
# Load results from JSON
with open("results.json", "r") as f:
    results = json.load(f)
# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Wheat Seeds Model Results</title>
</head>
<body>
<h1>Wheat Seeds Model Results</h1>
<p><strong>Accuracy:</strong> {results['accuracy']:.2f}</p>
<h2>Feature Importances:</h2>
<ul>
<li>Area: {results['feature_importances'][0]:.2f}</li>
<li>Perimeter: {results['feature_importances'][1]:.2f}</li>
<li>Compactness: {results['feature_importances'][2]:.2f}</li>
<li>Length: {results['feature_importances'][3]:.2f}</li>
<li>Width: {results['feature_importances'][4]:.2f}</li>
<li>Asymmetry: {results['feature_importances'][5]:.2f}</li>
<li>Groove: {results['feature_importances'][6]:.2f}</li>
</ul>
</body>
</html>
"""
# Save HTML to a file
with open("index.html", "w") as f:
    f.write(html_content)
print("HTML file generated: index.html")
```

5. Run the above Python script, `generate_html.py`, in the VS Code terminal window.
6. Verify that `index.html` is generated with the model results.

---

## Task 3: Automating CI/CD with GitHub actions

This task aims to setup a CI/CD pipeline that pushes the files to GitHub and deploys them on the GitHub pages.

1. In the `.github/workflows` folder, create a file called `ci.yml`, containing the following.

```yaml
name: CI/CD for Hosting Wheat Seeds HTML App

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:

      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt

      - name: Train the model
        run: |
          source venv/bin/activate
          python train_model.py

      - name: Generate HTML
        run: |
          source venv/bin/activate
          python generate_html.py

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./
```

2. Create a `requirements.txt` file in the project folder with the following dependencies in it.

```
pandas
scikit-learn
matplotlib
```

3. Commit and push changes to the GitHub repository.

```bash
git add .
git commit -m "Set up CI/CD pipeline for GitHub Pages"
git push origin main
```

4. To enable GitHub pages, go to the GitHub repository, select Settings and under Pages, set the source to the Deploy from a branch and the branch to main /root, then click Save.
5. To view the deployed app, click on Visit site or navigate to your GitHub page URL (e.g., http://your-username.github.io/wheat-seeds-html-ci-cd/).

---

## Lab review

1. What is the purpose of the Deploy to GitHub Pages step in the CI/CD workflow?

A. To train the machine learning model

B. To push the model’s accuracy and feature importance to GitHub

C. To publish the index.html file to GitHub Pages

D. To configure GitHub repository secrets

STOP

You have completed this lab.
