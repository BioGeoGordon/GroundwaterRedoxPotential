# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:52:18 2024

@author: Gordo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os

redox_couple = 'All_O2_H2O'

# Specify your output directory here
output_directory = f'/user/directory/out/path/{redox_couple}/'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the data
file_path = 'user/directory/input/path/GSS_All_Charge_less10pct_O2.csv'
data = pd.read_csv(file_path, header=[0, 1])

# Constants
R = 8.3145  # J/mol/K
F = 96485  # C/mol
n_e = 4  # Number of electrons
n_h = 4  # Number of protons
T_k = 273.15  # Kelvin offset
molar_mass_O2 = 31.999  # g/mol

# Add a Sample ID column
data.insert(0, 'Sample ID', range(1, len(data) + 1))

# Preprocessing and Calculations
# Example calculations (Replace these with your actual calculations)
data['TpH'] = (data['Temperature']['C'] + T_k) * data['pH']['pH']

## Why is IAP not accounting for H2O stoichiometry?
data['IAP'] = np.log(data['O2(aq)']['activity']) - np.log(data['Activity of H2O']['activity'])
# data['IAP'] = np.log((data['O2(aq)']['activity'])/(data['Activity of H2O']['activity']))
data['log_K (in situ)'] = 93.7 - 0.3318 * data['Temperature']['C'] + 0.001056 * (data['Temperature']['C']**2) - 0.00000234 * (data['Temperature']['C']**3) + 0.000000002363 * (data['Temperature']['C']**4)
data['E0 (in situ)'] = (R * (data['Temperature']['C'] + T_k) / (n_e * F)) * np.log(10) * data['log_K (in situ)']
data['E_pH'] = -(n_h / n_e) * ((R * (data['Temperature']['C'] + T_k) / F) * np.log(10) * data['pH']['pH'])
data['E_a'] = (R * (data['Temperature']['C'] + T_k) / (n_e * F)) * data['IAP']
data['E_c'] = (R * (data['Temperature']['C'] + T_k) / (n_e * F)) * np.log((data['O2(aq)']['mg/l'] / (molar_mass_O2 * 1000)))
data['E_a_rxn'] = data['E0 (in situ)'] + data['E_pH'] + data['E_a']
data['E_c_rxn'] = data['E0 (in situ)'] + data['E_pH'] + data['E_c']
data['E-EoTdivT'] = (data['E_a_rxn']-data['E0 (in situ)'])/(data['Temperature']['C'] + T_k)
data['Error_GWB_a'] = data['E_a_rxn'] - data['Eh (O2(aq) /H2O )']['V']

## TpH
data['E-EoTdivT'] = (data['E_a_rxn']-data['E0 (in situ)'])/(data['Temperature']['C'] + T_k)
data['Error_act_cont'] = data['E_c_rxn'] - data['E_a_rxn']


# Save the calculated data
calculations_path = os.path.join(output_directory, f'{redox_couple}_calculations.csv')
data.to_csv(calculations_path, index=False)

all_stats = []

# Define a function for running regression and exporting the results
def run_linear_regression_and_export(X, y, variable_name, output_dir, adjust_predictions=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate standard errors
    residuals = y_train - y_pred_train
    SSE = np.sum(residuals**2)
    SE_slope = sqrt(SSE / (len(X_train) - 2)) / sqrt(np.sum((X_train - X_train.mean())**2))
    SE_intercept = SE_slope * sqrt(np.mean(X_train**2))
    
    # Calculate R2 before changing how TpH model calculates MAE, MSE, RMSE
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    
    # Adjust predictions if needed
    if adjust_predictions:
        y_pred_train = (model.coef_[0] * X_train.values.flatten() + model.intercept_) * (data.loc[X_train.index][('Temperature', 'C')] + T_k) + data.loc[X_train.index]['E0 (in situ)']
        y_pred_test = (model.coef_[0] * X_test.values.flatten() + model.intercept_) * (data.loc[X_test.index][('Temperature', 'C')] + T_k) + data.loc[X_test.index]['E0 (in situ)']
        y_train = data.loc[X_train.index]['E_a_rxn']
        y_test = data.loc[X_test.index]['E_a_rxn']


    # Calculate metrics and convert units to mV
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train) 
    rmse_train = sqrt(mse_train) 

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = sqrt(mse_test)

    # Save metrics to dictionary
    if variable_name == 'E_a_rxn':
        model_name = 'pH'
    else:
        model_name = 'TpH'
    
    stats = {
        'Model': f'{model_name} Train',
        'Count Train': len(y_train),
        'Count Test': len(y_test),
        'Slope (mV)': model.coef_[0] * 1000,
        'Intercept (mV)': model.intercept_ * 1000,
        'Alpha (mv)': (model.coef_[0]*7 + model.intercept_) * 1000,
        'SE Slope (mV)': SE_slope * 1000,
        'SE Intercept (mV)': SE_intercept * 1000,
        'MAE Train (mV)': mae_train* 1000,
        'MSE Train (mV^2)': mse_train* 1000000,
        'RMSE Train (mV)': rmse_train* 1000,
        'R2 Train': r2_train,
        'MAE Test (mV)': mae_test* 1000,
        'MSE Test (mV^2)': mse_test* 1000000,
        'RMSE Test (mV)': rmse_test* 1000,
        'R2 Test': r2_test
    }
    all_stats.append(stats)
    
    # Export detailed datasets
    train_df = pd.DataFrame({'Sample ID': X_train.index + 1, 'Predicted': y_pred_train, 'Actual': y_train})
    test_df = pd.DataFrame({'Sample ID': X_test.index + 1, 'Predicted': y_pred_test, 'Actual': y_test})
    train_df.to_csv(os.path.join(output_dir, f'{redox_couple}_{model_name}_training_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, f'{redox_couple}_{model_name}_testing_data.csv'), index=False)


# Run models
for variable in ['E_a_rxn', 'E-EoTdivT']:
    adjust = variable == 'E-EoTdivT'
    run_linear_regression_and_export(data[['pH']], data[variable], variable, output_directory, adjust_predictions=adjust)

# Export all statistics to a single file
summary_df = pd.DataFrame(all_stats)
summary_path = os.path.join(output_directory, f'{redox_couple}_model_stats_summary.csv')
summary_df.to_csv(summary_path, index=False)
