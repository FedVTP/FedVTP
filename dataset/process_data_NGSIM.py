import pandas as pd
import numpy as np


def ExtractUniqueId(carData, i):
    ID = carData.iloc[0, 0].copy() + i*10000
    print(str(ID))
    carData['Vehicle_ID'] = carData['Vehicle_ID'].map(lambda x: x + i * 10000).copy()
    carData['Global_Time'] = carData['Global_Time'].map(lambda x: x % 100000000).copy()
    carData = carData[['Global_Time', 'Vehicle_ID', 'Local_X', 'Local_Y', 'Lane_ID']]
    carData = carData[carData.Global_Time % 200 == 0]
    return carData


def main():
    data = pd.read_csv("us101-0750am-0805am_smoothed_21.csv", header=None)
    data.columns = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y', 'Global_X',
                    'Global_Y',
                    'v_length', 'v_Width', 'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID', 'Preceding', 'Following',
                    'Space_Headway',
                    'Time_Headway']
    i = 0
    finally_data = pd.DataFrame()
    for ID in set(data.Vehicle_ID):
        ID_data = data[data.Vehicle_ID == ID].sort_values(by='Global_Time')
        ID_data.reset_index(drop=True, inplace=True)
        unique_id_data = ExtractUniqueId(ID_data, i)
        finally_data = pd.concat([finally_data, unique_id_data], axis=0)
    finally_data.to_csv('NGSIM/rawdata/train/' + str(1) + '/us101' + "_" + str(i) + '.csv',
                index=False, header=False)


if __name__ == '__main__':
    main()
