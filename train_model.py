import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load cleaned data
df = pd.read_csv('data/cleaned_data.csv')

# Split data into features and target variable
# Assuming 'target' is the column in your dataset that indicates heart disease presence
X = df.drop('target', axis=1)  # Features
y = df['target']                # Target variable

# Step 1: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train Model
# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Step 3: Hyperparameter Tuning
# Define the parameter grid for tuning
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Step 4: Evaluate Model
# Predict on test data
y_pred = best_model.predict(X_test)

# Measure performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Step 5: Save Model
joblib.dump(best_model, 'ml_model/heart_disease_model.pkl')
print("Model saved to ml_model/heart_disease_model.pkl")