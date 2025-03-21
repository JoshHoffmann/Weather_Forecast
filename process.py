import pandas as pd
import numpy as np
import plotting




def GetData():
    '''Get pre processed data set for modelling'''

    master_df = GetRawData()
    master_df = FullRange(master_df)
    master_df = EWMFill(master_df)
    master_df = FeatureAveraging(master_df)

    plotting.PlotTimeSeriesFromMaster(master_df, 'avg feature Data')
    s1 = master_df[master_df['Station']=='63rd Street Weather Station']
    s2 = master_df[master_df['Station'] == 'Foster Weather Station']
    s3 = master_df[master_df['Station'] == 'Oak Street Weather Station']
    return master_df

def GetRawData()->pd.DataFrame:
    '''Read and format raw data from csv'''

    data = pd.read_csv('data.csv')
    data['Measurement Timestamp'] = pd.to_datetime(data['Measurement Timestamp'], format='mixed')
    values = ['Air Temperature', 'Wet Bulb Temperature', 'Humidity', 'Total Rain', 'Wind Direction', 'Wind Speed',
              'Maximum Wind Speed','Barometric Pressure', 'Solar Radiation']

    data = data.pivot(index='Measurement Timestamp', values=values, columns='Station Name')

    stations = {station: data.xs(station, level='Station Name', axis=1) for station in data.columns.get_level_values(1).unique()}
    station_63rd = stations.get('63rd Street Weather Station')
    station_Foster = stations.get('Foster Weather Station')
    station_Oak = stations.get('Oak Street Weather Station')

    master_df = pd.concat(
        [df.assign(Station=station) for station, df in stations.items()], axis = 0
    ).reset_index()
    print(master_df.head())



    return master_df

def FullRange(master_df):

    date_range = pd.date_range(master_df['Measurement Timestamp'].min(), end =master_df['Measurement Timestamp'].max(),
                               freq='h')
    stations_date_filled = {}

    for station, station_df in master_df.groupby('Station'):
        station_df = station_df.set_index('Measurement Timestamp')
        station_df = station_df.reindex(date_range)
        station_df.index.name = 'Measurement Timestamp'
        station_df['Station'] = station

        stations_date_filled[station] = station_df.reset_index()
    master_df = pd.concat(stations_date_filled.values(), ignore_index=True)
    return master_df

def EWMFill(master):
    master = master.copy()
    features = master.drop(columns=['Station', 'Measurement Timestamp']).columns
    for feature in features:
        for station, station_df in master.groupby('Station'):
            valid_index = station_df[feature].dropna().index
            valid_times = station_df.loc[valid_index, 'Measurement Timestamp']
            gaps = valid_times.diff().dropna()
            max_gap = gaps.max().days*24
            if feature != 'Total Rain':
                span = max_gap
            else:
                span = 2
            master.loc[master['Station'] == station, feature] = (
                station_df[feature].fillna(station_df[feature].ewm(span=span).mean())
            )
    return master

def FeatureAveraging(master):
    features = ['Wet Bulb Temperature','Total Rain', 'Wind Speed', 'Maximum Wind Speed', 'Solar Radiation']
    master = master.copy()
    for feature in features:
        if feature in ['Wet Bulb Temperature', 'Total Rain', 'Solar Radiation']:
            station1 = master[master['Station']=='63rd Street Weather Station'][feature]
            station2 = master[master['Station']=='Oak Street Weather Station'][feature]
            avg = np.nanmean([station1.values, station2.values], axis=0)
            master.loc[master['Station']=='Foster Weather Station',feature] = avg
        else:
            station1 = master[master['Station'] == 'Foster Weather Station'][feature]
            station2 = master[master['Station'] == 'Oak Street Weather Station'][feature]
            avg = np.nanmean([station1.values, station2.values], axis=0)
            master.loc[master['Station'] == '63rd Street Weather Station', feature] = avg

    return master