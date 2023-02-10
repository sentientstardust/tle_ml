#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:31:27 2023

@author: becca
"""

import numpy as np
import pandas as pd
from Read_TLE_File import Read_TLE_File
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

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

# Split target satellite data in train/test
seg1 = 500 * len(CanX2_df) // 5250
CanX2_df_train = CanX2_df.iloc[:seg1]
CanX2_df_test = CanX2_df.iloc[seg1:]

# Combine dataframes
TLE_df_train = pd.concat([CanX2_df_train, SciSat_df, Odin_df], axis=1)
TLE_df_test = pd.concat([CanX2_df_test, SciSat_df, Odin_df], axis=1)

# Define training and testing dataframes
df_train = TLE_df_train.dropna()
df_test = TLE_df_test.dropna()

# Define target satellite
targsatdf = CanX2_df
targsatname = 'CanX2'

# Define multiple target columns
target_cols = [f'{targsatname} - Inclination', f'{targsatname} - RAAN', f'{targsatname} - B*',
               f'{targsatname} - Eccentricity', f'{targsatname} - Argument of Perigee',
               f'{targsatname} - Mean Anomaly', f'{targsatname} - Semi-Major Axis']

# Create an empty dataframe to store the error metrics
metrics_df = pd.DataFrame(columns = target_cols, index = ['RMSE', 'R2'])

for target_col in target_cols:
    
    # Define feature columns
    feature_cols = TLE_df_train.columns.drop(target_col)
    
    X_train = df_train[feature_cols]
    y_train = df_train[[target_col]].values.ravel()
    X_test = df_test[feature_cols]
    y_test = df_test[[target_col]].values.ravel()
    
    best_error = float('inf')
    best_errors = []
    best_models = {}
    min_difference = float('inf')
    
    models_and_predictions = {'RandomForestRegressor': (rf, y_pred_rf), 
                              'LinearRegression': (lr, y_pred_lr), 
                              'Ridge': (ridge, y_pred_ridge), 
                              'Lasso': (lasso, y_pred_lasso), 
                              'ElasticNet': (elnet, y_pred_elnet)}

    for i in range(-5, 5):
        # Initialize and fit the linear regression model
        rf = RandomForestRegressor(n_estimators=100)
        lr = LinearRegression()
        ridge = Ridge(alpha=10**(i))#cubesat
        lasso = Lasso(alpha=10**(i))#cubesat
        elnet = ElasticNet(alpha=10**(i), l1_ratio = 0.75)
    
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        lasso.fit(X_train, y_train)
        elnet.fit(X_train, y_train)

        # Make predictions on test set
        y_pred_rf = rf.predict(X_test)
        y_pred_lr = lr.predict(X_test)
        y_pred_ridge = ridge.predict(X_test)
        y_pred_lasso = lasso.predict(X_test)
        y_pred_elnet = elnet.predict(X_test)

        # Calculate error metrics
        mse_rf = r2_score(y_test, y_pred_rf)
        mse_lr = r2_score(y_test, y_pred_lr)
        mse_ridge = r2_score(y_test, y_pred_ridge)
        mse_lasso = r2_score(y_test, y_pred_lasso)
        mse_elnet = r2_score(y_test, y_pred_elnet)
        # mse_rf = mean_squared_error(y_test, y_pred_rf)
        # mse_lr = mean_squared_error(y_test, y_pred_lr)
        # mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        # mse_lasso = mean_squared_error(y_test, y_pred_lasso)
        # mse_elnet = mean_squared_error(y_test, y_pred_elnet)
    
        errors = [mse_rf, mse_lr, mse_ridge, mse_lasso, mse_elnet]
        # min_error = min(errors)
        # closest_value = None
        for value in errors:
            difference = abs(1 - value)
            if difference < min_difference:
                min_difference = difference
                closest_value = value
        best_errors.append(closest_value)
    
    # if best_errors.index(min(best_errors)):
        
    # best_error = min(best_errors)
    # print(best_error)    
    # if min(errors)==mse_rf:
    #     rmse = np.sqrt(mse_rf)
    #     r2 = r2_score(y_test, y_pred_rf)
    #     y_pred = y_pred_rf
    #     print(target_col)
    #     print('model: rf')
    # elif min(errors)==mse_lr:
    #     rmse = np.sqrt(mse_lr)
    #     r2 = r2_score(y_test, y_pred_lr)
    #     y_pred = y_pred_lr
    #     print(target_col)
    #     print('model: lr')
    # elif min(errors)==mse_ridge:
    #     rmse = np.sqrt(mse_ridge)
    #     r2 = r2_score(y_test, y_pred_ridge)
    #     y_pred = y_pred_ridge
    #     print(target_col)
    #     print('model: ridge')
    # elif min(errors)==mse_lasso:
    #     rmse = np.sqrt(mse_lasso)
    #     r2 = r2_score(y_test, y_pred_lasso)
    #     y_pred = y_pred_lasso
    #     print(target_col)
    #     print('model: lasso')
    # elif min(errors)==mse_elnet:
    #     rmse = np.sqrt(mse_elnet)
    #     r2 = r2_score(y_test, y_pred_elnet)
    #     y_pred = y_pred_elnet
    #     # y_pred_test = rf_ridge2.predict(X_test_test)
    #     print(target_col)
    #     print('model: elnet')
    
#     # Append the error metrics to the dataframe
#     metrics_df.loc['RMSE', target_col] = rmse
#     metrics_df.loc['R2', target_col] = r2

#     # Plot predicted vs actual values
#     X_train_time = np.linspace(0, (seg1/30), len(y_train))
#     X_test_time = np.linspace((seg1/30), (6750/30), len(y_test))  # start time for test data after training data
#     plt.figure(figsize=(12.5,5))
#     plt.plot(X_train_time, y_train, label='Actual (Training)', color = 'red')
#     plt.plot(X_test_time, y_test, label='Actual (Test)', color = 'blue')
#     plt.plot(X_test_time, y_pred, label='Predicted', color = 'green')
#     plt.xlabel('Time (months)')
#     plt.ylabel(target_col)
#     plt.title(f'Predicted vs Actual {target_col}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    

# print(metrics_df)


# fintime= datetime.datetime.now()

# print(fintime - starttime)