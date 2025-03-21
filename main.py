import pandas as pd
import numpy as np
import process
import matplotlib.pyplot as plt
import modelling
import feature_extract
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

def main():
    master_df = process.GetData() # Retrieve cleaned data
    master_in, master_out = SplitData(master_df)

    f = master_in[master_in['Station']=='Foster Weather Station']

    master_in = feature_extract.GetFeatures(master_in)
    master_out = feature_extract.GetFeatures(master_out)
    modelling.XGBModel(master_in,master_out)

    return None

def SplitData(master):
    in_sample = {}
    out_sample = {}
    i_split = int(0.8*len(master['Measurement Timestamp'].unique()))
    for station, station_df in master.groupby('Station'):
        s_in, s_out = station_df.iloc[:i_split], station_df.iloc[i_split:]
        in_sample[station] = s_in
        out_sample[station] = s_out


    master_in_sample = pd.concat(in_sample, ignore_index=True)
    master_out_sample = pd.concat(out_sample, ignore_index=True)
    return master_in_sample, master_out_sample

if __name__ == '__main__':
    main()
    plt.show()