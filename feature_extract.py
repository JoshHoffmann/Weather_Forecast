import pandas as pd
import numpy as np
import plotting
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def GetFeatures(master_df):

    def PCAFeatures():
        master = master_df.copy()
        stations = {}
        features = ['Humidity','Wind Direction', 'Wind Speed', 'Barometric Pressure', 'Solar Radiation']

        for station, station_df in master.groupby('Station'):
            pca_features = np.zeros((len(station_df),2))

            for t in range(1,len(station_df)):
                past_data = station_df.iloc[:t+1].copy()

                scaler = StandardScaler()
                past_data[features] = scaler.fit_transform(past_data[features])

                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(past_data[features])

                pca_features[t,:] = pca_result[-1,:]

            for i in range(2):
                station_df[f'PCA_{i+1}'] = pca_features[:,i]

            stations[station] = station_df
        master = pd.concat(stations,ignore_index=True)
        return master
    def CreateTarget():
        master = master_df.copy()
        stations = {}
        for station, station_df in master.groupby('Station'):
            station_df.insert(1, 'Air_Temp_Target', station_df['Air Temperature'].shift(-1).dropna())
            stations[station] = station_df
        master = pd.concat(stations, ignore_index=True)
        return master

    def SelectFeatures():
        master = master_df.copy()
        stations = {}
        to_drop = ['Humidity','Wind Direction', 'Wind Speed', 'Barometric Pressure', 'Solar Radiation',
                   'Maximum Wind Speed']
        for station, station_df in master.groupby('Station'):
            station_df = station_df.drop(columns=to_drop)
            stations[station] = station_df
        master = pd.concat(stations, ignore_index=True)
        return master


    master_df = CreateTarget()

    master_df = PCAFeatures()
    master_df = SelectFeatures()

    plotting.PlotLag1Correlations(master_df, 'test')
    plotting.PlotScatters(master_df)
    return master_df
