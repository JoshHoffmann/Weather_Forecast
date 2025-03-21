import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from itertools import product

import plotting


def XGBModel(master:pd.DataFrame,master_out):

    eta_space = np.linspace(0.01,0.1,10)
    depth_space = [1,2,3,4]
    features_list = ['Air Temperature', 'Wet Bulb Temperature',
       'Total Rain', 'PCA_1', 'PCA_2']
    station_names = master['Station'].unique()

    eta_params = {s: None for s in station_names}
    depth_params = {s: None for s in station_names}
    validation_mse = {s: None for s in station_names}

    models = {s: None for s in station_names}
    forecast_dict = {s:None for s in station_names}

    for station, station_df in master.groupby('Station'):
        target = station_df['Air_Temp_Target'].dropna()
        features = station_df[features_list].loc[target.index]

        eta, depth, val_mse = ParamGridSearch(target,features, eta_space, depth_space)
        eta_params[station], depth_params[station], validation_mse[station] = eta, depth, val_mse

        print(f'eta_params = {eta_params}')
        print(f'depth_params = {depth_params}')
        print(f'Cross Validation min mse = {val_mse}')
        # Now train on full in sample data
        model = xgb.XGBRegressor(n_estimators=200, eta=eta, max_depth=depth)
        fitted_model = model.fit(features, target)
        models[station] = fitted_model

    for station, station_df in master_out.groupby('Station'):

        ### Now train on full in sample data and forecast out of sample
        features_out = station_df[features_list]

        fitted_model = models[station]
        forecasts_out = fitted_model.predict(features_out)
        forcast_df = pd.DataFrame({'Forecast':forecasts_out,'Station':station}, index = features_out.index)
        forcast_df = forcast_df.shift(1).dropna()
        forecast_dict[station] = forcast_df
    master_forecast = pd.concat(forecast_dict, ignore_index=True)
    plotting.PlotForecasts(master_out,master_forecast)


    return


def ParamGridSearch(target:pd.Series, features:pd.DataFrame,eta_space,depth_space):
    '''Perform grid search over parameter spaces to determine mse minimising params'''

    best_mse, best_lags, best_eta, best_depth = np.inf, None, None, None

    for eta, depth in product(eta_space,depth_space):
        print(f'eta = {eta}, depth = {depth}')

        mse = ValidateXGB(target, features,eta,depth)
        if mse< best_mse:
            best_mse, best_eta, best_depth = mse, eta, depth

    return best_eta, best_depth, best_mse

def ValidateXGB(series:pd.Series,features:pd.DataFrame, eta:float,max_depth:int):
    '''Function to time series cross validate at fixed parameters and features. Returns average out-sample mse over folds.'''
    ts_split = TimeSeriesSplit(n_splits=5)
    mse_list = []
    for i_train, i_test in ts_split.split(series):
        X_train, Y_train = features.iloc[i_train], series.iloc[i_train]
        X_test, Y_test = features.iloc[i_test], series.iloc[i_test]

        model = xgb.XGBRegressor(n_estimators=200, eta=eta, max_depth=max_depth)
        fitted_model = model.fit(X_train,Y_train)

        Y_forecast = fitted_model.predict(X_test)
        Y_forecast = pd.Series(Y_forecast, index = Y_test.index)
        mse = mean_squared_error(Y_test,Y_forecast)
        mse_list.append(mse)
    avg_mse = np.mean(mse_list)

    return avg_mse