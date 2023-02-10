#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:11:08 2023

@author: becca
"""

import numpy as np
import pandas as pd
from Read_TLE_File import Read_TLE_File
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import matplotlib.pyplot as plt

# Start timer
starttime= datetime.datetime.now()

# Disable column display limit
pd.set_option('display.max_columns', None)

# Import TLE data
CanX2_df = Read_TLE_File('/Users/becca/Desktop/cubesat/fresh start <3/satellite data/CanX-2 - 100-5350') # CanX-2, day 100, delta 5250, 06/08/08 to 23/12/22
SciSat_df = Read_TLE_File('/Users/becca/Desktop/cubesat/fresh start <3/satellite data/CanX-29.txt') # SciSat, day 100, delta 6750, 21/11/03 to 15/05/22
Odin_df = Read_TLE_File('/Users/becca/Desktop/cubesat/fresh start <3/satellite data/Odin - 100-7850') # Odin, day 100, delta 7750, 31/05/01 to 19/08/22

# Define a function to drop unnecessary columns
def dropcol(TLE_df):
    TLE_df_drop = TLE_df.drop(['Sat No.', 'International Designator', '2nd Derivative of Mean Motion / 6', 'Revolution Number'], axis = 1)
    # Use time delta
    TLE_df_drop = TLE_df_drop.drop(['Epoch Year', 'Epoch Day', 'Datetime'], axis = 1)
    return TLE_df_drop

# Drop unnecessary columns
CanX2_df = dropcol(CanX2_df)
SciSat_df = dropcol(SciSat_df)
Odin_df = dropcol(Odin_df)

# Define a function to scale the data
def MinMax(df):
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_array, columns = df.columns)
    return df

# Scale data
CanX2_df = MinMax(CanX2_df)
SciSat_df = MinMax(SciSat_df)
Odin_df = MinMax(Odin_df)

# Adjust column titles
CanX2_df.columns = [f'CanX2 - {col}' for col in CanX2_df.columns]
SciSat_df.columns = [f'SciSat - {col}' for col in SciSat_df.columns]
Odin_df.columns = [f'Odin - {col}' for col in Odin_df.columns]

seg1 = 500 * len(CanX2_df) // 5250
CanX2_df_train = CanX2_df.iloc[:seg1]
CanX2_df_test = CanX2_df.iloc[seg1:]

# Combine dataframes
TLE_df_train = pd.concat([CanX2_df_train, SciSat_df, Odin_df], axis=1)
TLE_df_test = pd.concat([CanX2_df_test, SciSat_df, Odin_df], axis=1)

# TLE_df = MinMax(TLE_df)

# Split dataframe into training and test sets
# seg1 = 500 * len(CanX2_df) // 5250
# seg2 = len(SciSat_df)
# df_train = TLE_df.iloc[:seg1].dropna()
# df_test = TLE_df.iloc[seg1:].dropna()
df_train = TLE_df_train.dropna()
df_test = TLE_df_test.dropna()

# Define target satellite
targsatdf = CanX2_df
targsatname = 'CanX2'
target_col = f"{targsatname} - B*"

# # Define target columns
# # target_cols = [targsat + 'Inclination', targsat + 'RAAN', targsat + 'B*', 
# #                targsat + 'Eccentricity', targsat + 'Argument of Perigee', 
# #                targsat + 'Mean Anomaly', targsat + 'Semi-Major Axis']


# Define feature and target columns
feature_cols = TLE_df_train.columns.drop(target_col)
print(feature_cols)

# Create an empty dataframe to store the error metrics
# metrics_df = pd.DataFrame(columns = target_col, index = ['RMSE', 'R2'])

X_train = df_train[feature_cols]
y_train = df_train[[target_col]].values.ravel()
X_test = df_test[feature_cols]
y_test = df_test[[target_col]].values.ravel()

print(X_test)

# Initialize and fit the linear regression model
ridge = Ridge(alpha=0.01)#cubesat
# ridge = Lasso(alpha=0.1)#cubesat
# lr = LinearRegression()
# rf = RandomForestRegressor(n_estimators=1000)
ridge.fit(X_train, y_train)
# rf.fit(X_train, y_train)

# Make predictions on test set
y_pred_lr = ridge.predict(X_test)
# y_pred_lr = rf.predict(X_test)

# Calculate error metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse_lr)
r2 = r2_score(y_test, y_pred_lr)
y_pred = y_pred_lr
    
# Append the error metrics to the dataframe
# metrics_df.loc['RMSE', target_col] = rmse
# metrics_df.loc['R2', target_col] = r2

# Plot predicted vs actual values
X_train_time = np.linspace(0, (seg1/30), len(y_train))
X_test_time = np.linspace((seg1/30), (6750/30), len(y_test))  # start time for test data after training data
plt.figure(figsize=(12.5,5))
plt.plot(X_train_time, y_train, label='Actual (Training)', color = 'red')
plt.plot(X_test_time, y_test, label='Actual (Test)', color = 'blue')
plt.plot(X_test_time, y_pred, label='Predicted', color = 'green')
plt.xlabel('Time (months)')
plt.ylabel(target_col)
plt.title(f'Predicted vs Actual {target_col}')
plt.legend()
plt.grid(True)
plt.show()
    

print(rmse, r2)

# print(metrics_df)


fintime= datetime.datetime.now()

print(fintime - starttime)
