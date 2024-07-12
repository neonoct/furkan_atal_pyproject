# Description: This script trains and evaluates Decision Tree, Random Forest, XGBoost, and a Voting Classifier with all o them on the music genre dataset.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from func import preprocess_data

# Load the dataset
df = pd.read_csv('./music_genre.csv')

# Preprocess the data
X, y = preprocess_data(df)

# Function to train and evaluate Decision Tree
def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Commented out the grid search as the best parameters are already known
    # param_grid = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 10, 20, 30, 40, 50],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    
    # grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)
    
    # Use the best parameters from the previous grid search
    best_tree_model = DecisionTreeClassifier(
        criterion='entropy', max_depth=10, min_samples_leaf=1, min_samples_split=10, random_state=42)
    
    start_time = time.time()
    best_tree_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred_tree = best_tree_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_tree))
    print(confusion_matrix(y_test, y_pred_tree))
    test_accuracy = accuracy_score(y_test, y_pred_tree)
    mae = mean_absolute_error(y_test, y_pred_tree)
    print(f'Test accuracy of Decision Tree: {test_accuracy}')
    print(f'MAE of Decision Tree: {mae}')
    print(f'Training time for Decision Tree: {training_time} seconds')

# Function to train and evaluate Random Forest
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Commented out the grid search as the best parameters are already known
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    
    # grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)
    
    # Use the best parameters from the previous grid search
    best_forest_model = RandomForestClassifier(
        max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300, random_state=42)
    
    start_time = time.time()
    best_forest_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred_forest = best_forest_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_forest))
    print(confusion_matrix(y_test, y_pred_forest))
    test_accuracy = accuracy_score(y_test, y_pred_forest)
    mae = mean_absolute_error(y_test, y_pred_forest)
    print(f'Test accuracy of Random Forest: {test_accuracy}')
    print(f'MAE of Random Forest: {mae}')
    print(f'Training time for Random Forest: {training_time} seconds')

# Function to train and evaluate XGBoost
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Commented out the grid search as the best parameters are already known
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 4, 5, 6, 7],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'subsample': [0.7, 0.8, 0.9, 1.0]
    # }
    
    # grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)
    
    # Use the best parameters from the previous grid search
    best_xgb_model = XGBClassifier(
        learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.7, random_state=42)
    
    start_time = time.time()
    best_xgb_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred_xgb = best_xgb_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_xgb))
    print(confusion_matrix(y_test, y_pred_xgb))
    test_accuracy = accuracy_score(y_test, y_pred_xgb)
    mae = mean_absolute_error(y_test, y_pred_xgb)
    print(f'Test accuracy of XGBoost: {test_accuracy}')
    print(f'MAE of XGBoost: {mae}')
    print(f'Training time for XGBoost: {training_time} seconds')

# Train and evaluate models
train_decision_tree(X, y)
train_random_forest(X, y)
train_xgboost(X, y)

# Define the individual models with the best parameters
decision_tree = DecisionTreeClassifier(
    criterion='entropy', max_depth=10, min_samples_leaf=1, min_samples_split=10, random_state=42)
random_forest = RandomForestClassifier(
    max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300, random_state=42)
xgboost = XGBClassifier(
    learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.7, random_state=42)

# Combine the models using a Voting Classifier
voting_model = VotingClassifier(
    estimators=[('dt', decision_tree), ('rf', random_forest), ('xgb', xgboost)],
    voting='soft')  # Use 'hard' for majority voting, 'soft' for weighted voting

# Split the data again for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Voting Classifier
start_time = time.time()
voting_model.fit(X_train, y_train)
training_time_voting = time.time() - start_time

# Predict and evaluate
y_pred_voting = voting_model.predict(X_test)
print(classification_report(y_test, y_pred_voting))
print(confusion_matrix(y_test, y_pred_voting))
test_accuracy_voting = accuracy_score(y_test, y_pred_voting)
mae_voting = mean_absolute_error(y_test, y_pred_voting)
print(f'Test accuracy of Voting Classifier: {test_accuracy_voting}')
print(f'MAE of Voting Classifier: {mae_voting}')
print(f'Training time for Voting Classifier: {training_time_voting} seconds')

# Draw confusion matrix
cm = confusion_matrix(y_test, y_pred_voting)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Voting Classifier')
plt.show()
