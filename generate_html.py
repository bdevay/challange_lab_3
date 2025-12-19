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
