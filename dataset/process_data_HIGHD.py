import pandas as pd
import math
import os


def resample10hz(df):
    df.reset_index(drop=True, inplace=True)
    offset = int(df.iloc[0, -1] % 100)
    first = pd.DataFrame()
    if offset == 0 :
        first = pd.DataFrame()
    elif offset % 40 == 0:
        df = df.iloc[2-int(offset/40):, :]
        tmp = df.iloc[0:2, :]
        df = df.iloc[3:, :]
        tmp.reset_index(drop=True, inplace=True)
        first = pd.DataFrame(0.5*tmp.iloc[0, :] + 0.5*tmp.iloc[1, :])
        first = first.T
        first.loc[0, 'Preceding'] = tmp.loc[1, 'Preceding']
        first.loc[0, 'Following'] = tmp.loc[1, 'Following']
        first.loc[0, 'Lane_ID'] = tmp.loc[1, 'Lane_ID']
    elif offset % 20 == 0:
        df = df.iloc[int((100-offset)/40):, :]
    df_time = df.set_index(pd.to_timedelta(df['Global_Time'], unit='ms'))
    t1 = df_time.iloc[:, [0, 2, 3, 6, 7, 8, 9, 14]].resample('100ms').interpolate(method='linear')
    t2 = df_time.iloc[:, [1, 4, 5, 10, 11, 12, 13]].resample('100ms').bfill()
    output = pd.concat([t1, t2], axis=1)
    output = pd.concat([first, output], axis=0)
    output = output[[ 'Frame_ID', 'Vehicle_ID','Local_Y', 'Local_X', 'v_length', 'v_Width', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 'Lane_ID', 'Preceding', 'Following',  'Total_Frames', 'Global_Time']]
    output.reset_index(drop=True, inplace=True)
    if output.iloc[-1, 0] == output.iloc[-2, 0]:
        output = output.iloc[:-1, :]
        if df.iloc[-1, -1] % 100 == 60:
            last = pd.DataFrame(0.5 * df.iloc[-2, :] + 0.5 * df.iloc[-3, :])
            last = last.T
            last.loc[0, 'Preceding'] = df.iloc[-3, 10]
            last.loc[0, 'Following'] = df.iloc[-3, 11]
            last.loc[0, 'Lane_ID'] = df.iloc[-3, 12]
            output = pd.concat([output, last], axis=0)
        elif df.iloc[-1, -1] % 100 == 20:
            last = pd.DataFrame(0.5 * df.iloc[-1, :] + 0.5 * df.iloc[-2, :])
            last = last.T
            last.loc[0, 'Preceding'] = df.iloc[-3, 10]
            last.loc[0, 'Following'] = df.iloc[-3, 11]
            last.loc[0, 'Lane_ID'] = df.iloc[-3, 12]
            output = pd.concat([output, last], axis=0)
    return output

def resample20hz(df):
    df.reset_index(drop=True, inplace=True)
    df = df[df.Global_Time % 200 == 0]
    return df

def main():
    path = 'data/raw/highD/'
    alldata = []
    t = pd.DataFrame()
    for i in range(6):
        alldata.append(t)
    for i in range(1, 61):
        if i < 10 :
            num = '0' + str(i) + '_'
        else:
            num = str(i) + '_'
        data = pd.read_csv(path + num + 'tracks.csv')
        metadata = pd.read_csv(path + num + 'tracksMeta.csv')
        roaddata = pd.read_csv(path + num + 'recordingMeta.csv')
        site = roaddata.loc[0, 'locationId']
        print("成功导入数据" + str(i))
        data.rename(columns={'id': 'Vehicle_ID', 'frame': 'Frame_ID', 'x': 'Local_Y', 'y': 'Local_X', 'width': 'v_Width',
                     'height': 'v_length', 'laneId': 'Lane_ID', 'precedingId': 'Preceding', 'followingId': 'Following',
                             'xVelocity': 'yVelocity', 'yVelocity': 'xVelocity', 'xAcceleration':'yAcceleration', 'yAcceleration':'xAcceleration'
                     }, inplace=True)
        del data['frontSightDistance']
        del data['backSightDistance']
        del data['dhw']
        del data['thw']
        del data['ttc']
        del data['precedingXVelocity']
        del data['leftPrecedingId']
        del data['leftAlongsideId']
        del data['leftFollowingId']
        del data['rightPrecedingId']
        del data['rightAlongsideId']
        del data['rightFollowingId']
        data['Total_Frames'] = 0
        data['Global_Time'] = 0
        idx = 0
        new_data = pd.DataFrame()
        for ID in set(data.Vehicle_ID):
            # print("处理中" + str(i) + "+"+ str(ID))
            tmp = metadata.loc[ID - 1, 'numFrames']
            front = metadata.loc[ID - 1, 'initialFrame']
            data.loc[idx:idx + tmp - 1, 'Total_Frames'] = tmp
            data.loc[idx:idx + tmp - 1, 'Vehicle_ID'] = ID + i * 10000
            for tidx in range(idx, idx + tmp):
                data.loc[tidx, 'Global_Time'] = (front-1) * 40 + i * 1000000000
                data.loc[tidx, 'Local_Y'] = data.loc[tidx, 'Local_Y'] / 0.3048
                data.loc[tidx, 'Local_X'] = data.loc[tidx, 'Local_X'] / 0.3048
                front += 1
            df = data.iloc[idx:idx + tmp, :]
            if len(df) < 200:
                print("丢弃" + str(ID))
                idx += tmp
                continue
            print("提取成功" + str(ID))
            # tmp_data = df.reset_index(drop=True, inplace=False)
            tmp_data = resample20hz(df)
            tmp_data['Global_X'] = 0
            tmp_data['Global_Y'] = 0
            tmp_data['v_Class'] = 0
            tmp_data['Space_Headway'] = 0
            tmp_data['Time_Headway'] = 0
            tmp_data['v_Vel'] = 0
            tmp_data['v_Acc'] = 0
            new_data = pd.concat([new_data, tmp_data], axis=0)
            idx += tmp

        finally_data = new_data[
            ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y', 'Global_X', 'Global_Y',
             'v_length', 'v_Width', 'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID', 'Preceding', 'Following', 'Space_Headway',
             'Time_Headway', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]
        finally_data = finally_data[['Global_Time', 'Vehicle_ID', 'Local_X', 'Local_Y', 'Lane_ID']]
        finally_data = finally_data.sort_values(by='Global_Time')
        finally_data.reset_index(drop=True, inplace=True)
        # save_data(finally_data, str(site))
        # 不同路段
        if i == 3:
            finally_data.to_csv('HIGHD/rawdata/val/' + str(site-1) + '/highd_site' + str(site) + "_" + str(i) + '.csv',
                                index=False, header=False)
        elif i == 6:
            finally_data.to_csv('HIGHD/rawdata/val/' + str(site-1) + '/highd_site' + str(site) + "_" + str(i) + '.csv',
                                index=False, header=False)
        elif i == 10:
            finally_data.to_csv('HIGHD/rawdata/val/' + str(site-1) + '/highd_site' + str(site) + "_" + str(i) + '.csv',
                                index=False, header=False)
        elif i in [22, 23, 24]:
            finally_data.to_csv('HIGHD/rawdata/val/' + str(site-1) + '/highd_site' + str(site) + "_" + str(i) + '.csv',
                                index=False, header=False)
        elif i in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]:
            finally_data.to_csv('HIGHD/rawdata/val/' + str(site-1) + '/highd_site' + str(site) + "_" + str(i) + '.csv',
                                index=False, header=False)
        elif i == 60:
            finally_data.to_csv('HIGHD/rawdata/val/' + str(site-1) + '/highd_site' + str(site) + "_" + str(i) + '.csv',
                                index=False, header=False)
        else :
            finally_data.to_csv(
                'HIGHD/rawdata/train/' + str(site-1) + '/highd_site' + str(site) + "_" + str(i) + '.csv',
                index=False, header=False)

if __name__ == '__main__':
    main()
