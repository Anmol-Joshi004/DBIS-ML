import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Function to load emails and labels from separate train and test paths
def load_data(path, label_indicator):
    emails, labels = [], []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            emails.append(file.read())
            labels.append(1 if label_indicator in filename else 0)  # 1 for spam, 0 for non-spam
    return emails, labels

# Paths to training and testing datasets
train_path = "./train_test_mails/train-mails"  # Correct path for training data
test_path = "./train_test_mails/test-mails"    # Correct path for testing data

# Load the train and test data separately
train_emails, train_labels = load_data(train_path, 'spm')
test_emails, test_labels = load_data(test_path, 'spm')

# Vectorize the text data separately for training and test sets
tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(train_emails)
X_test = tfidf.transform(test_emails)
y_train = pd.Series(train_labels)
y_test = pd.Series(test_labels)

# Define models to evaluate with hyperparameter grids
models = {
    'Naive Bayes': (MultinomialNB(), {}),
    'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 10]}),
    'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]})
}

# Store results for each model
results = {}
confusion_matrices = {}

# Train and evaluate each model with Grid Search for hyperparameter tuning
for model_name, (model, params) in models.items():
    print(f"Training {model_name}...")
    
    # Use GridSearchCV for hyperparameter tuning with cross-validation
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Predict on training data and calculate training accuracy
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Calculate test accuracy and classification report on test data
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate misclassification error
    misclassification_error = 1 - accuracy

    # Save the results
    results[model_name] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': accuracy,
        'misclassification_error': misclassification_error,
        'classification_report': class_report
    }

    # Compute and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[model_name] = cm

# Display organized results
print("\nModel Performance Summary:")
print(f"{'Model':<20} {'Train Accuracy':<15} {'Test Accuracy':<15} {'Misclassification Error':<25}")
print("=" * 75)

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['train_accuracy']:.4f}          {metrics['test_accuracy']:.4f}          {metrics['misclassification_error']:.4f}")

# Identify the best model based on test accuracy
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_model = models[best_model_name][0]  # Get the model object

print(f"\nBest Model: {best_model_name} with Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# Save the best model and TF-IDF vectorizer for future use
with open("best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Visualization of model performance
model_names = list(results.keys())
train_accuracies = [results[name]['train_accuracy'] for name in model_names]
test_accuracies = [results[name]['test_accuracy'] for name in model_names]
misclassification_errors = [results[name]['misclassification_error'] for name in model_names]

# Set up bar chart for performance comparison
x = np.arange(len(model_names))

# Create figure and axis for performance comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Plot training accuracy
bars_train = ax.bar(x - 0.2, train_accuracies, width=0.2, label='Training Accuracy', color='b', align='center')
# Plot test accuracy
bars_test = ax.bar(x, test_accuracies, width=0.2, label='Test Accuracy', color='g', align='center')
# Plot misclassification error
bars_error = ax.bar(x + 0.2, misclassification_errors, width=0.2, label='Misclassification Error', color='r', align='center')

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1 for better scaling
ax.legend()

# Adding grid lines for better readability
ax.yaxis.grid(True, linestyle='--', which='both', linewidth=0.7)
ax.set_axisbelow(True)  # Ensure grid lines are below the bars

# Annotate bars with their respective values
for bar in bars_train:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom')

for bar in bars_test:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom')

for bar in bars_error:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom')

# Show the performance plot
plt.tight_layout()
plt.show()

# Visualization of confusion matrices
fig, axes = plt.subplots(nrows=len(model_names), ncols=1, figsize=(10, 4 * len(model_names)))
fig.suptitle('Confusion Matrices', fontsize=16)

# Plot each confusion matrix
for ax, model_name in zip(axes, model_names):
    cm = confusion_matrices[model_name]
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.5)
    ax.set_title(model_name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Annotate confusion matrix values
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, value, ha='center', va='center')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Spam', 'Spam'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-Spam', 'Spam'])

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()