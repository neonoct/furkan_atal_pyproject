# Description: This script implements the k-Nearest Neighbors (kNN) algorithm to classify music genres. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from func import preprocess_data
import time

# Load the dataset
df = pd.read_csv('./music_genre.csv')

X, y = preprocess_data(df)

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the kNN model
knn_model = KNeighborsClassifier(n_neighbors=15)

# Fit the model
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Calculate accuracy
test_acc_knn = accuracy_score(y_test, y_pred)
print(f'Test accuracy of kNN: {test_acc_knn}')

# Calculate MAE on the test set without PCA
mae_knn = mean_absolute_error(y_test, y_pred)
print(f'MAE on test set (without PCA): {mae_knn}')

# Training time calculation for k-NN without PCA
start_time = time.time()
knn_model.fit(X_train, y_train)
training_time_knn = time.time() - start_time
print(f'Training time for k-NN (without PCA): {training_time_knn} seconds')

# Try different values of k
k_values = range(1, 21)
accuracies = []

# Different values of k and distance metrics to test
metrics = ['euclidean', 'manhattan', 'chebyshev']

# Dictionary to store the accuracy results
results = {}

for metric in metrics:
    accuracies = []
    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    results[metric] = accuracies

# Plot the accuracy for different k values and metrics
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(k_values, results[metric], marker='o', label=f'Metric: {metric}')
plt.title('kNN Accuracy for Different k Values and Distance Metrics')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Show total rows and columns before PCA 
print(X_train.shape)

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=10)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Show total rows and columns after PCA
print(X_train_pca.shape)

# Define the kNN model with PCA-transformed data
knn_model_pca = KNeighborsClassifier(n_neighbors=15)
knn_model_pca.fit(X_train_pca, y_train)

# Predict on the test set
y_pred_pca = knn_model_pca.predict(X_test_pca)

# Calculate accuracy
test_acc_knn_pca = accuracy_score(y_test, y_pred_pca)
print(f'Test accuracy of kNN with PCA: {test_acc_knn_pca}')

# Calculate MAE on the test set with PCA
mae_knn_pca = mean_absolute_error(y_test, y_pred_pca)
print(f'MAE on test set (with PCA): {mae_knn_pca}')

# Training time calculation for k-NN with PCA
start_time = time.time()
knn_model_pca.fit(X_train_pca, y_train)
training_time_knn_pca = time.time() - start_time
print(f'Training time for k-NN (with PCA): {training_time_knn_pca} seconds')

# Function to apply PCA and evaluate kNN
def evaluate_knn_with_pca(n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    knn_model_pca = KNeighborsClassifier(n_neighbors=15)
    knn_model_pca.fit(X_train_pca, y_train)
    
    y_pred_pca = knn_model_pca.predict(X_test_pca)
    return accuracy_score(y_test, y_pred_pca)

# Test different numbers of PCA components
components_range = range(5, X_train.shape[1] + 1)
accuracies_pca = [evaluate_knn_with_pca(n) for n in components_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(components_range, accuracies_pca, marker='o')
plt.title('kNN Accuracy with Different Numbers of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Define the kNN model with the best parameters found so far
best_knn_model = KNeighborsClassifier(n_neighbors=15, metric='manhattan', weights='distance')

# Perform cross-validation
cv_scores = cross_val_score(best_knn_model, X_scaled, y, cv=5)

# Calculate and print the average accuracy
average_cv_accuracy = cv_scores.mean()
print(f'Average cross-validated accuracy of kNN: {average_cv_accuracy}')
