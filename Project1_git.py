# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:52:33 2024

@author: beros_p4ca
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score, f1_score,accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import joblib

#Step 1 : Data Processing
df = pd.read_csv("Project_1_Data.csv") #Reading the provided CSV file 

#Preview of first 10 rows in data set
print(df.head(10))

#Preforming basic statistical analysis and summarizing the results
print("\nStatistical Analaysis of Data Set")
print(df.describe())

#Step 2: Data Visualization
#Plotting a histogram of the dataset 
df.hist(bins =25, figsize=(20,15))
plt.savefig("Histogram_Plot.png")
plt.show()

#Plotting a 3D figure of the dataset as a scatter plot and linegraph 
ax = plt.axes(projection="3d") #function to create a 3D grid
x= df["X"]
y= df["Y"]
z= df["Z"]
s=df["Step"]

sc=ax.scatter(x,y,z,c=s, marker='o',cmap='viridis' ) #plotting scatter plot with verfied parameters

ax.set_xlabel('X') #defining x label
ax.set_ylabel('Y') #defining y label
ax.set_zlabel('Z') #defining z label
ax.set_title("XYZ 3D Plot") #defining title of graph

plt.colorbar(sc,label='Step') #adding colour bar indicating the step values

plt.show() #displaying plot

X_train = df[['X','Y','Z','Step']]

#Step 3: Correlation Analysis 
# Selecting the coordinate features (X, Y, Z)
coords = ['X', 'Y', 'Z']

# Standardizing the coordinate features
scaler = StandardScaler()
scaled_coords = scaler.fit_transform(df[coords])  # Fit and transform the selected features

# Creating a DataFrame for scaled coordinates
scaled_coords_df = pd.DataFrame(scaled_coords, columns=coords)

# Calculate the correlation matrix for the scaled coordinates
corr_matrix = scaled_coords_df.corr()  # Compute the correlation matrix
print("\nCorrelation Matrix:")
print(corr_matrix)

# Plotting the heatmap for the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")  # function to create heatmap
plt.title('Correlation Matrix of Scaled Coordinates')
plt.show()

#Step 4 : Classification Model Development/Engineering 
#Principal Component Analysis Transformation
pca = PCA(n_components=2)  # Reduce to 2 components
principal_components = pca.fit_transform(scaled_coords)  # Fit and transform the scaled coordinates

# Creating a new dataframe for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Adding the principal components to the original dataframe (including the target variable 'Step')
df_pca = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

#Calculating the correlation matrix for PCA components and 'Step'
corr_matrix_pca = df_pca[['PC1', 'PC2', 'Step']].corr()  # Calculate correlation with 'Step'
print("\nCorrelation Matrix After PCA:")
print(corr_matrix_pca)

# Plotting the correlation matrix for PCA components and 'Step'
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_pca, annot=True, cmap="coolwarm", fmt=".2f")  # Create a heatmap with annotations
plt.title('Correlation Matrix After PCA')
plt.show()

# Splitting the Data
# Separating the features (PC1, PC2) and the target (Step)
X = df_pca[['PC1', 'PC2']]  # Features
y = df_pca['Step']           # Target variable

# Splitting the data into training sets and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Classification Models
rf = RandomForestClassifier(random_state=42) #define random forest model
knn = KNeighborsClassifier() #define KNeighbors model
svm = SVC(random_state=42) #define SVC model

# Define the hyperparameters grid for each model
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Perform grid search for each model
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=1)
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy', verbose=1)
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy', verbose=1)

# Fitting the models to the training data
grid_search_rf.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)
grid_search_svm.fit(X_train, y_train)

# Output the best model for each
print("\nBest Random Forest Model:", grid_search_rf.best_estimator_)
print("\nBest KNN Model:", grid_search_knn.best_estimator_)
print("\nBest SVM Model:", grid_search_svm.best_estimator_)

#Step 5: Model Performance 
# Get the best models from GridSearchCV
best_rf = grid_search_rf.best_estimator_  # Best Random Forest model
best_knn = grid_search_knn.best_estimator_  # Best KNN model
best_svm = grid_search_svm.best_estimator_  # Best SVM model

# Make predictions on the test set using each of the best models
rf_predictions = best_rf.predict(X_test)  # Predictions from Random Forest
knn_predictions = best_knn.predict(X_test)  # Predictions from KNN
svm_predictions = best_svm.predict(X_test)  # Predictions from SVM

# Calculate evaluation metrics for Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)  # Calculate accuracy
rf_precision = precision_score(y_test, rf_predictions, average='weighted')  # Calculate precision
rf_f1_score = f1_score(y_test, rf_predictions, average='weighted')  # Calculate F1 score

# Calculate evaluation metrics for KNN model
knn_accuracy = accuracy_score(y_test, knn_predictions)  # Calculate accuracy
knn_precision = precision_score(y_test, knn_predictions, average='weighted')  # Calculate precision
knn_f1_score = f1_score(y_test, knn_predictions, average='weighted')  # Calculate F1 score

# Calculate evaluation metrics for SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)  # Calculate accuracy
svm_precision = precision_score(y_test, svm_predictions, average='weighted')  # Calculate precision
svm_f1_score = f1_score(y_test, svm_predictions, average='weighted')  # Calculate F1 score

# Compile the performance metrics into a DataFrame for easy comparison
performance_metrics = pd.DataFrame({
    'Model': ['Random Forest', 'KNN', 'SVM'],  # List of models
    'Accuracy': [rf_accuracy, knn_accuracy, svm_accuracy],  # Accuracy for each model
    'Precision': [rf_precision, knn_precision, svm_precision],  # Precision for each model
    'F1 Score': [rf_f1_score, knn_f1_score, svm_f1_score]  # F1 score for each model
})

# Display the performance metrics
print(performance_metrics)

# Create a confusion matrix for the best model (e.g., Random Forest)
conf_matrix = confusion_matrix(y_test, rf_predictions)  # Compute confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)  # Prepare display for confusion matrix
disp.plot(cmap=plt.cm.Blues)  # Plot the confusion matrix using a blue color map
plt.title('Confusion Matrix for Random Forest Model')  # Title for the confusion matrix plot
plt.show()  # Show the plot

#Step 6: Stacked Model Performance Analysis
# Define the base estimators using the best models from grid search
estimators = [
    ('rf', best_rf),  # Best Random Forest model
    ('knn', best_knn)  # Best KNN model
]

# Create the stacking classifier using the base estimators and SVM as the final estimator
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=best_svm)

# Fit the stacking classifier on the training
stacking_clf.fit(X_train, y_train)

# Make predictions on the test set using the stacking classifier
stacking_predictions = stacking_clf.predict(X_test)

# Calculate evaluation metrics for the stacking classifier
stacking_accuracy = accuracy_score(y_test, stacking_predictions)  # Calculate accuracy
stacking_precision = precision_score(y_test, stacking_predictions, average='weighted')  # Calculate precision
stacking_f1_score = f1_score(y_test, stacking_predictions, average='weighted')  # Calculate F1 score

# Compile the performance metrics for the stacking classifier into a DataFrame
stacking_performance = pd.DataFrame({
    'Model': ['Stacking Classifier'],  # Model name
    'Accuracy': [stacking_accuracy],  # Accuracy for stacking classifier
    'Precision': [stacking_precision],  # Precision for stacking classifier
    'F1 Score': [stacking_f1_score]  # F1 score for stacking classifier
})

# Display the performance metrics for the stacking classifier
print("\nPerformance Metrics for Stacking Classifier:")
print(stacking_performance)

# Create a confusion matrix for the stacking classifier
stacking_conf_matrix = confusion_matrix(y_test, stacking_predictions)  # Compute confusion matrix
stacking_disp = ConfusionMatrixDisplay(confusion_matrix=stacking_conf_matrix)  # Prepare display for confusion matrix
stacking_disp.plot(cmap=plt.cm.Blues)  # Plot the confusion matrix using a blue color map
plt.title('Confusion Matrix for Stacking Classifier')  # Title for the confusion matrix plot
plt.show()  # Show the plot

# Step 6.1: Performance Comparison with Base Models
print("\nPerformance Comparison:")
print(performance_metrics)  # Display base models' performance metrics
print(stacking_performance)  # Display stacking classifier's performance metrics

# Analyze the impact of stacking
if stacking_accuracy > max(performance_metrics['Accuracy']):
    print("\nThe Stacking Classifier shows improved accuracy compared to individual models.")
    print("This improvement can be attributed to the complementary strengths of the base models.")
else:
    print("\nThe Stacking Classifier's accuracy is comparable to individual models.")
    print("This indicates limited effectiveness of stacking for this dataset.")
    
#Step 7: Model Evaluation
joblib_file = "stacking_classifier_model.joblib"
joblib.dump(stacking_clf, joblib_file)  # Save the stacking classifier

print(f"Model saved as {joblib_file}")
# Predict maintenance steps based on new coordinates
# Define the new coordinates for prediction
new_coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

# Standardize the new coordinates using the same scaler used for training
scaled_new_coords = scaler.transform(new_coordinates)
new_coordinates_pca = pca.transform(scaled_new_coords)

# Predict the corresponding maintenance steps using the loaded model
predicted_steps = stacking_clf.predict(new_coordinates_pca)

# Output the predictions
for coords, step in zip(new_coordinates, predicted_steps):
    print(f"Coordinates: {coords} => Predicted Maintenance Step: {step}")
