import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def PlotTimeSeriesFromMaster(master,title):
    stations = master['Station'].unique()
    features = master.drop(columns=['Station','Measurement Timestamp']).columns
    print(features)
    for f in features:
        figure, ax = plt.subplots(nrows=len(stations),ncols=1)
        figure.suptitle(f'{title}: {f}')
        figure.subplots_adjust(hspace=1)
        for i, station in enumerate(stations):
            x = master[master['Station']==station]['Measurement Timestamp']
            y = master[master['Station']==station][f]
            ax[i].plot(x,y)
            ax[i].set_xlabel('Datetime')
            ax[i].set_ylabel(f)
            ax[i].set_title(station)

def PlotLag1Correlations(master,title):
    for station, station_df in master.groupby('Station'):
        print(station_df.head())
        station_df = station_df.drop(columns=['Measurement Timestamp','Station'])
        print(station_df.head())
        plt.figure()
        corr = station_df.corr()
        sn.heatmap(corr)
        plt.title(station)

def PlotScatters(master,title='Feature scatters'):
    stations = master['Station'].unique()
    features = master.drop(columns=['Station', 'Measurement Timestamp']).columns
    print(features)
    for f in features:
        figure, ax = plt.subplots(nrows=len(stations), ncols=1)
        figure.suptitle(f'{title}: {f}')
        figure.subplots_adjust(hspace=1)
        for i, station in enumerate(stations):
            y = master[master['Station'] == station]['Air Temperature'].shift(-1).dropna()
            x = master[master['Station']==station][f].loc[y.index]
            sn.scatterplot(x=x,y=y,ax=ax[i], alpha=0.7,edgecolor=None)
            ax[i].set_xlabel(f)
            ax[i].set_ylabel('Lag -1 Air Temp')
            ax[i].set_title(station)
def PlotForecasts(master, master_forecast,title=''):
    stations = master['Station'].unique()
    figure, ax = plt.subplots(nrows=len(stations), ncols=1)
    figure.suptitle(f'{title}')
    figure.subplots_adjust(hspace=1)
    for i, station in enumerate(stations):
        y_f = master_forecast[master_forecast['Station'] == station]['Forecast']
        x = master[master['Station'] == station]['Measurement Timestamp'].iloc[:len(y_f.index)]
        y_obs = master[master['Station'] == station]['Air Temperature'].iloc[:len(y_f.index)]

        ax[i].plot(x, y_obs,label='Observed')
        ax[i].plot(x, y_f, label = 'Forecast')
        ax[i].set_xlabel('Datetime')
        ax[i].set_ylabel('Air Temperature')
        ax[i].set_title(station)
        ax[i].legend()
