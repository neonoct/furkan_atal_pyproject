# SVM model for music genre classification
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from func import preprocess_data

# Load the dataset
df = pd.read_csv('./music_genre.csv')

X, y = preprocess_data(df)

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVM model with default parameters
svm_model = SVC()

# Measure training time
start_time = time.time()
svm_model.fit(X_train, y_train)
training_time_svm = time.time() - start_time

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)

# Calculate accuracy
test_acc_svm = accuracy_score(y_test, y_pred_svm)
print(f'Test accuracy of SVM: {test_acc_svm}')

# Calculate MAE
mae_svm = mean_absolute_error(y_test, y_pred_svm)
print(f'MAE of SVM: {mae_svm}')

# Commented out the GridSearchCV part
# Define the parameter grid for hyperparameter tuning
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto']
# }

# Initialize the GridSearchCV object
# grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=2, n_jobs=-1)

# Measure training time for grid search
# start_time = time.time()
# grid_search.fit(X_train, y_train)
# training_time_grid_search = time.time() - start_time

# Print the best parameters and best score
# print(f'Best parameters: {grid_search.best_params_}')
# print(f'Best cross-validation accuracy: {grid_search.best_score_}')

# Use the best parameters found from GridSearchCV
best_svm_model = SVC(C=10, gamma='scale', kernel='rbf')

# Measure training time for the best SVM model
start_time = time.time()
best_svm_model.fit(X_train, y_train)
training_time_best_svm = time.time() - start_time

# Predict on the test set using the best SVM model
y_pred_best_svm = best_svm_model.predict(X_test)

# Calculate accuracy
test_acc_best_svm = accuracy_score(y_test, y_pred_best_svm)
print(f'Test accuracy of best SVM: {test_acc_best_svm}')

# Calculate MAE for the best SVM model
mae_best_svm = mean_absolute_error(y_test, y_pred_best_svm)
print(f'MAE of best SVM: {mae_best_svm}')

print(f'Training time for best SVM: {training_time_best_svm} seconds')

# One-vs-One (OvO) strategy
ovo_model = OneVsOneClassifier(SVC(C=10, gamma='scale', kernel='rbf'))

# Measure training time for OvO
start_time = time.time()
ovo_model.fit(X_train, y_train)
training_time_ovo = time.time() - start_time

# Predict on the test set using OvO
y_pred_ovo = ovo_model.predict(X_test)

# Calculate accuracy for OvO
test_acc_ovo = accuracy_score(y_test, y_pred_ovo)
print(f'Test accuracy of OvO SVM: {test_acc_ovo}')

# Calculate MAE for OvO
mae_ovo = mean_absolute_error(y_test, y_pred_ovo)
print(f'MAE of OvO SVM: {mae_ovo}')

# One-vs-All (OvA) strategy
ova_model = OneVsRestClassifier(SVC(C=10, gamma='scale', kernel='rbf'))

# Measure training time for OvA
start_time = time.time()
ova_model.fit(X_train, y_train)
training_time_ova = time.time() - start_time

# Predict on the test set using OvA
y_pred_ova = ova_model.predict(X_test)

# Calculate accuracy for OvA
test_acc_ova = accuracy_score(y_test, y_pred_ova)
print(f'Test accuracy of OvA SVM: {test_acc_ova}')

# Calculate MAE for OvA
mae_ova = mean_absolute_error(y_test, y_pred_ova)
print(f'MAE of OvA SVM: {mae_ova}')

# Print training times
print(f'Training time for SVM: {training_time_svm} seconds')
# print(f'Training time for Grid Search SVM: {training_time_grid_search} seconds')
print(f'Training time for OvO SVM: {training_time_ovo} seconds')
print(f'Training time for OvA SVM: {training_time_ova} seconds')
