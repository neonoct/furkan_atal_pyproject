# Description: This script demonstrates how to use a voting classifier with a neural network, SVM, and XGBoost model to classify music genres.
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from func import preprocess_data

# Load the dataset
df = pd.read_csv('./music_genre.csv')

X, y = preprocess_data(df)

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the best neural network model function
def create_best_model(optimizer='rmsprop', neurons=128, dropout_rate=0.3):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neurons, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neurons, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Even there are many parts in the code written using LLMs This is taken directly from Chatgpt
#to be able to use the KerasClassifier with integer labels instead of one-hot encoded labels
# Custom wrapper for KerasClassifier to handle integer labels
class MyKerasClassifier(KerasClassifier):
    def fit(self, X, y, **kwargs):
        return super().fit(X, to_categorical(y), **kwargs)
    
    def predict(self, X, **kwargs):
        pred = super().predict(X, **kwargs)
        return np.argmax(pred, axis=1)

# Create the KerasClassifier for the best neural network
best_nn_model = MyKerasClassifier(model=create_best_model, optimizer='rmsprop', neurons=128, dropout_rate=0.3, epochs=100, batch_size=64, verbose=2)

# Define the SVM and XGBoost models
svm_model = SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42)
xgb_model = XGBClassifier(learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.7, random_state=42)

# Function to measure training time, accuracy, and MAE
def evaluate_model(voting_model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    voting_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = voting_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return test_accuracy, mae, training_time

# Neural Network + SVM
voting_model_nn_svm = VotingClassifier(estimators=[('nn', best_nn_model), ('svm', svm_model)], voting='soft')
test_acc_nn_svm, mae_nn_svm, training_time_nn_svm = evaluate_model(voting_model_nn_svm, X_train, y_train, X_test, y_test)
print(f'Test accuracy of Neural Network + SVM Voting Classifier: {test_acc_nn_svm}')
print(f'MAE of Neural Network + SVM Voting Classifier: {mae_nn_svm}')
print(f'Training time for Neural Network + SVM Voting Classifier: {training_time_nn_svm} seconds')

# Neural Network + XGBoost
voting_model_nn_xgb = VotingClassifier(estimators=[('nn', best_nn_model), ('xgb', xgb_model)], voting='soft')
test_acc_nn_xgb, mae_nn_xgb, training_time_nn_xgb = evaluate_model(voting_model_nn_xgb, X_train, y_train, X_test, y_test)
print(f'Test accuracy of Neural Network + XGBoost Voting Classifier: {test_acc_nn_xgb}')
print(f'MAE of Neural Network + XGBoost Voting Classifier: {mae_nn_xgb}')
print(f'Training time for Neural Network + XGBoost Voting Classifier: {training_time_nn_xgb} seconds')

# Neural Network + XGBoost + SVM
voting_model_nn_xgb_svm = VotingClassifier(estimators=[('nn', best_nn_model), ('xgb', xgb_model), ('svm', svm_model)], voting='soft')
test_acc_nn_xgb_svm, mae_nn_xgb_svm, training_time_nn_xgb_svm = evaluate_model(voting_model_nn_xgb_svm, X_train, y_train, X_test, y_test)
print(f'Test accuracy of Neural Network + XGBoost + SVM Voting Classifier: {test_acc_nn_xgb_svm}')
print(f'MAE of Neural Network + XGBoost + SVM Voting Classifier: {mae_nn_xgb_svm}')
print(f'Training time for Neural Network + XGBoost + SVM Voting Classifier: {training_time_nn_xgb_svm} seconds')

# Print classification report and confusion matrix for the best model
best_model = voting_model_nn_xgb_svm  # Assuming this is the best model; adjust as necessary
y_pred_best = best_model.predict(X_test)
print(f'Classification Report for the Best Model:\n{classification_report(y_test, y_pred_best)}')
print(f'Confusion Matrix for the Best Model:\n{confusion_matrix(y_test, y_pred_best)}')
