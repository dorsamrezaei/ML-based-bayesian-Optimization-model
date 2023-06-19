import pandas as pd
import numpy as np
from numpy import mean
import matplotlib as mp
import matplotlib.pyplot as plt
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_validate, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDRegressor, SGDClassifier 
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE #Recursive Feature Elimination (RFE)_method of feature selection
from keras import regularizers
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from process_data import *
from sys import exit
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import make_scorer, mean_squared_error

x, y, x_original = process_data()
x, x_val, y, y_val = train_test_split(x,y,test_size = 0.15, random_state = 42)

from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.01)
# Define the model architecture (NOT Being USED)
def estimator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(44, activation='tanh'))
   # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(120, activation='tanh', kernel_regularizer=regularizers.l2(0.01))) # used to be 0.05
    #model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='linear'))
    # Compile the model
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Initialize lists to store training and test MSE for each fold
train_mse_list = []
test_mse_list = []
train_r2_list=[]
test_r2_list=[]
train_mae_list=[]
test_mae_list=[]
models = []
# Define the number of folds for cross-validation
num_folds = 10
# Create the KFold object
kf = KFold(n_splits=num_folds, shuffle=True)
x_dataframe=x.copy(deep=True)
# Loop over the folds
for train_index, test_index in kf.split(x):
    
    # Split the data into training and test sets for this fold
    x_fold_train=x.iloc[train_index]
    y_fold_train = y[train_index]
    x_fold_test=x.iloc[test_index]
    y_fold_test = y[test_index]
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    x_fold_train = scaler.fit_transform(x_fold_train)
    x_fold_test = scaler.transform(x_fold_test)
    #x_fold_train.iloc[:,0:29] = scaler.fit_transform(x_fold_train.iloc[:,0:29])
    #x_fold_test.iloc[:,0:29] = scaler.fit_transform(x_fold_test.iloc[:,0:29])
    y_fold_train = scaler.fit_transform(y_fold_train.reshape(-1, 1))
    y_fold_train = np.ravel(y_fold_train)
    y_fold_test = scaler.transform(y_fold_test.reshape(-1, 1))
    y_fold_test = np.ravel(y_fold_test)
    
    # Train the model on the training set for this fold
    # Use KerasRegressor wrapper (from Keras to sklearn)
    # The packages we use are meant to be run with sklearn models
    model = KerasRegressor(build_fn=estimator, epochs=1000, batch_size=5, verbose=0)                                                                                                                          
    history=model.fit(x_fold_train, y_fold_train, validation_data=(x_fold_test, y_fold_test))
    
    # Evaluate the model on the training set for this fold
    train_pred = model.predict(x_fold_train)
    train_mse = np.mean((train_pred - y_fold_train)**2)
    train_mse_list.append(train_mse)
    # Calculate the R2 score for the predictions
    r2_train = r2_score(y_fold_train, train_pred)
    train_r2_list.append(r2_train)
    mae_train=np.mean(np.abs(train_pred - y_fold_train))
    train_mae_list.append(mae_train)
 
    # Evaluate the model on the test set for this fold
    test_pred = model.predict(x_fold_test)
    test_mse = np.mean((test_pred - y_fold_test)**2)
    test_mse_list.append(test_mse)
    r2_test = r2_score(y_fold_test, test_pred)
    test_r2_list.append(r2_test)
    mae_test=np.mean(np.abs(test_pred - y_fold_test))
    test_mae_list.append(mae_test)
    
    # Save the trained model for this fold
    models.append(model)
    
# model.save('my_model.h5')
#model.model.save('saved_reg_model_final.h5')

# Select the best model based on validation set performance
best_model_index = np.argmax(test_r2_list)
best_model = models[best_model_index]

# Compute the average training and test MSE across all folds
avg_train_mse = sum(train_mse_list) / num_folds
avg_test_mse = sum(test_mse_list) / num_folds
avg_train_r2 = sum(train_r2_list) / num_folds
avg_test_r2 = sum(test_r2_list) / num_folds
avg_test_mae = sum(test_mae_list) / num_folds
avg_train_mae = sum(train_mae_list) / num_folds

# Print the results
print(f'Average training set MSE: {avg_train_mse}')
print(f'Average test set MSE: {avg_test_mse}')
print(f'Average training set R2: {avg_train_r2}')
print(f'Average test set R2: {avg_test_r2}')
print(f'Average train set mae: {avg_train_mae}')
print(f'Average test set mae: {avg_test_mae}')

#val set evaluation, not seen by model
x = scaler.fit_transform(x,y)
x_val = scaler.transform(x_val)
y = scaler.fit_transform(y.reshape(-1, 1))
y_val= scaler.transform(y_val.reshape(-1, 1))
y_val = np.ravel(y_val)
val_pred = best_model.predict(x_val)
val_mse = np.mean((val_pred - y_val)**2)
val_r2=r2_score(y_val,val_pred)
val_mae=np.mean(np.abs(val_pred-y_val))

print(f'Average val set R2: {val_r2}')
print(f'Average val set mae: {val_mae}')
print(f'Average val set mae: {val_mae}')

# create a dictionary of data
data = {
    'val_pred': val_pred,
    'y_val': y_val,
    'r2': val_r2,
    'mse': val_mse,
    'mae': val_mae
    }
# create a pandas dataframe
df_val_results= pd.DataFrame(data)
# Export the DataFrame to an Excel file
df_val_results.to_excel('validation set_results.xlsx', index=False)

# Plot the test set MSE, MAE and R2 values with standard deviations across the folds fot test set
mse_mean = avg_test_mse
mse_std = np.std(test_mse_list)
mae_mean = avg_test_mae
mae_std = np.std(test_mae_list)
r2_mean = avg_test_r2
r2_std = np.std(test_r2_list)
# Define the data for the test set plot
test_mse_mean = avg_test_mse
test_mse_std = np.std(test_mse_list)
test_mae_mean= avg_test_mae
test_mae_std = np.std(test_mae_list)
test_r2_mean = avg_test_r2
test_r2_std = np.std(test_r2_list)
# Define the data for the training set plot
train_mse_mean = avg_train_mse
train_mse_std = np.std(train_mse_list)
train_mae_mean = avg_train_mae
train_mae_std = np.std(train_mae_list)
train_r2_mean = avg_train_r2
train_r2_std = np.std(train_r2_list)

# Set the plot parameters
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(3)
width = 0.35
labels = ['MSE', 'MAE', 'R2']
# Define the data and standard deviation values for the training set
train_data = [train_mse_mean, train_mae_mean, train_r2_mean]
train_std = [train_mse_std,train_mae_std,train_r2_std]
# Define the data and standard deviation values for the test set
test_data = [val_mse, val_mae, val_r2]
test_std = [0,0,0]
# Plot the performance metrics for the training set with error bars
ax.bar(x_pos - width/2, train_data, width, align='center', alpha=0.5, color='darkgreen', label='Train Set', yerr=train_std, capsize=10)
# Plot the performance metrics for the test set with error bars
ax.bar(x_pos + width/2, test_data, width, align='center', alpha=0.5, color='darkblue', label='Test Set', capsize=10)
# Set the plot labels and title
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Performance')
ax.set_title('Model Performance for Classification Model')
ax.legend()
plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
plt.show()
