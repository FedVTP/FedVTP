# -*- coding: utf-8 -*-

# //$URL:: https://github.com/Rim-El-Ballouli/NGSIM-dataset-smoothing/tree/master/smothing-code
# //$Author:: Rim El Ballouli
# //$Date:: 1/02/2021

"""
This code
    1. smoothes out the noise in the local x and local y values in the NGSIM Dataset
    2. recomputes the velocites and accelerations
    3. saves smoothed dataset to three separate csv files
"""

from scipy import signal
import pandas as pd
import numpy as np
import numexpr



def get_smoothed_x_y_vel_accel(dataset, window):
    """
    this function returns four numpy arrays representing the smoothed
    1) local x, 2) local y, 3) velocity, 4) acceleration for a given numpy dataset.
    It relies on two helper functions  get_smoothed_x_y and get_smoothed_vel_accel
    :param dataset: numpy array representing the dataset to smooth it's local X , Y, velocity, acceleration
                    The numpy array should contains info for a single vehicle ID
                    otherwise result smoothed values are incorrect
    :param window: a smoothing window must be an odd integer value
                    if it set to 11 this means points are smoothed with 1 second interval equivalent to 10 points
                    if it set to 21 this means points are smoothed with 2 second interval equivalent to 20 points
    """
    smoothed_x_values, smoothed_y_values = get_smoothed_x_y(dataset, window)

    initial_vel = dataset[0, 11]
    initial_accel = dataset[0, 12]

    time_values = dataset[:, time_column]
    smoothed_vel, smoothed_accel, smoothed_vel_x, smoothed_vel_y, smoothed_accel_x, smoothed_accel_y = get_smoothed_vel_accel(smoothed_x_values, smoothed_y_values,
                                                          time_values, initial_vel, initial_accel)
    return smoothed_x_values, smoothed_y_values, smoothed_vel, smoothed_accel, smoothed_vel_x, smoothed_vel_y, smoothed_accel_x, smoothed_accel_y


def get_smoothed_x_y(dataset, window):
    """
    this function computes the smoothed local x and local y using savgol_filter for a given numpy dataset
    and returns two numpy arrays containing the smoothed x and y values.
    :param dataset: numpy array representing the dataset to smooth it's local X , Y, velocity, acceleration
                    The numpy array should contains info for a single vehicle ID
                    otherwise result smoothed values are incorrect
    :param window: a smoothing window must be an odd integer value
                    if it set to 11 this means points are smoothed with 1 second interval equivalent to 10 points
                    if it set to 21 this means points are smoothed with 2 second interval equivalent to 20 points
    """
    smoothed_x_values = signal.savgol_filter(dataset[:, local_x], window, 3)
    smoothed_y_values = signal.savgol_filter(dataset[:, local_y], window, 3)

    return smoothed_x_values, smoothed_y_values


def get_smoothed_vel_accel(smoothed_x_values, smoothed_y_values, time_values, initial_vel, initial_accel):
    """
    This function recomputes the velocity and acceleration for a given array of smoothed x, y values, time value
    To speedup calculation we use matrix functions to compute the values. For example, to compute velocity ,
    the x and y values are stacked to form matrix A. Then matrix B is then formed from Matrix A, but skipping t
    he first row. This implies that the x, y in first row in matrix B, are the next values of x and y in
    first row of matrix A. With two matrixes containing the current x, y and next x, y values we use fast matrix
    expressions to compute the smoothed velocities

    The function returns two numpy arrays representing the smoothed velocity and acceleration;

    :param smoothed_x_values: a numpy array of smoothed x values
    :param smoothed_y_values: a numpy array of smoothed y values
    :param time_values: a numpy array of smoothed time values values for the given x and y
    :param initial_vel: a single number containing the initial velocity
    :param initial_accel: a single number containing the initial acceleration
    """
    #create matrix of A containing current x and y and matrix B containing next x and y values
    x_y_matrix_A = np.column_stack((smoothed_x_values, smoothed_y_values))
    x_matrix_A = smoothed_x_values
    y_matrix_A = smoothed_y_values
    x_y_matrix_B = x_y_matrix_A [1:, :]
    x_matrix_B = x_matrix_A[1:]
    y_matrix_B = y_matrix_A[1:]
    #remove last row as it has no next values
    x_y_matrix_A = x_y_matrix_A[0:-1, :]
    x_matrix_A = x_matrix_A[0:-1]
    y_matrix_A = y_matrix_A[0:-1]

    # compute distance travelled between current and next x, y values
    dist_temp = numexpr.evaluate('sum((x_y_matrix_B - x_y_matrix_A)**2, 1)')
    dist = numexpr.evaluate('sqrt(dist_temp)')
    dist_x = numexpr.evaluate('x_matrix_B - x_matrix_A')
    dist_y = numexpr.evaluate('y_matrix_B - y_matrix_A')

    # create matrix A containing current time values, and matrix B containing next time values
    t_matrix_A = time_values
    t_matrix_B = t_matrix_A [1:]
    # remove last row
    t_matrix_A = t_matrix_A[0:-1]

    # evaluate smoothed velocity by dividing distance over delta time
    vel = numexpr.evaluate('dist * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_velocities = np.insert(vel, 0, initial_vel, axis=0)
    vel_x = numexpr.evaluate('dist_x * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_velocities_x = np.insert(vel_x, 0, vel_x[0], axis=0)
    vel_y = numexpr.evaluate('dist_y * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_velocities_y = np.insert(vel_y, 0, vel_y[0], axis=0)

    # create matrix A containing current velocities and matrix B containing next velocities
    vel_matrix_A = smoothed_velocities
    vel_x_matrix_A = smoothed_velocities_x
    vel_y_matrix_A = smoothed_velocities_y
    vel_matrix_B = vel_matrix_A [1:]
    vel_x_matrix_B = vel_x_matrix_A[1:]
    vel_y_matrix_B = vel_y_matrix_A[1:]
    # remove last row
    vel_matrix_A = vel_matrix_A[0:-1]
    vel_x_matrix_A = vel_x_matrix_A[0:-1]
    vel_y_matrix_A = vel_y_matrix_A[0:-1]

    # compute smoothed acceleration by dividing the delta velocity over delta time
    acc = numexpr.evaluate('(vel_matrix_B - vel_matrix_A) * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_accelaration = np.insert(acc, 0, initial_accel, axis=0)
    acc_x = numexpr.evaluate('(vel_x_matrix_B - vel_x_matrix_A) * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_accelaration_x = np.insert(acc_x, 0, acc_x[0], axis=0)
    acc_y = numexpr.evaluate('(vel_y_matrix_B - vel_y_matrix_A) * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_accelaration_y = np.insert(acc_y, 0, acc_y[0], axis=0)

    return np.array(smoothed_velocities), np.array(smoothed_accelaration), np.array(smoothed_velocities_x),  np.array(smoothed_velocities_y), np.array(smoothed_accelaration_x), np.array(smoothed_accelaration_y)


def smooth_dataset(window, train, file_names):
    """
    this function loops over a set of train data, and set of unique vehicle ids
    and for each vehicle id in each training dataset, it requests from helper methods the smoothed
    x, y, vel, accel values and replaces the old values with the smoothed values. Finally the new
    smoothed dataset is printed to a file
    :param dataset:  data frame representing the dataset to smooth it's local X and Y
    :param train: a list of 3 numpy arrays containing the original ngsim data
    """
    # find  unique vehicle ids in all the datasets, in the previous version
    vehicle_ids = train['Vehicle_ID'].unique()

    # convert to numpy arrays to fascilitate matrix operations to compute velocity and acceleration

    print(f"##### smoothing x, y, vel, accl values in train data")


    # for each unique vehicle id smooth x and y, recompute vel and acel
    # for vehicle in vehicle_ids:
    #     # 删除重复数据
    #     # print("处理重复数据中" + str(vehicle.item()))
    #     ID_data_index = train[train.Vehicle_ID == vehicle].index
    #     del_list = []
    #     for idx in ID_data_index[:-1]:
    #         if train.loc[idx + 1, 'Global_Time'] == train.loc[idx, 'Global_Time']:
    #             del_list.append(idx)
    # train.drop(index=del_list, inplace=True)
    # train.reset_index(drop=True, inplace=True)
    numpy_train = train.to_numpy()

    for vehicle in vehicle_ids:
        # 修正同一ID不同车辆
        print("处理数据ID" + str(vehicle.item()))
        lst_c = []
        ID_data_index = train[train.Vehicle_ID == vehicle].index
        for idx in ID_data_index:
            try :
                # if train.loc[idx + 1, 'Total_Frames'] != train.loc[idx, 'Total_Frames']:
                #     lst_c.append(idx + 1)
                if train.loc[idx + 1, 'Global_Time'] != train.loc[idx, 'Global_Time']+100 :
                    lst_c.append(idx + 1)
            except Exception as e :
                lst_c.append(idx+1)


        lst_c.insert(0, ID_data_index[0])
        lst_c.sort()
        #对于highd
        # lst_c = []
        # lst_c.append(ID_data_index[0])
        # lst_c.append(ID_data_index[-1])
        for j in range(len(lst_c) - 1):
            # create a filter for given vehicle id and use it to create a numpy array containing info only for that vehicle
            # filter = numpy_train[:,0] == vehicle
            # numpy_vehicle_dataset = numpy_train
            filter = np.zeros((len(numpy_train)), dtype=bool)
            filter[lst_c[j]:lst_c[j+1]] = True

            numpy_vehicle_dataset = numpy_train[filter]

            smoothed_x_values, smoothed_y_values, smoothed_vel,smoothed_accel, smoothed_vel_x, smoothed_vel_y, smoothed_accel_x, smoothed_accel_y = \
                 get_smoothed_x_y_vel_accel(numpy_vehicle_dataset, window)

            # replace values of x, y, vel, accel, with new smoothed values
            numpy_train[filter, local_x] = [x for x in smoothed_x_values]
            numpy_train[filter, local_y] = [x for x in smoothed_y_values]
            numpy_train[filter, v_vel] = [x for x in smoothed_vel]
            numpy_train[filter, v_acc] = [x for x in smoothed_accel]
            numpy_train[filter, xVelocity] = [x for x in smoothed_vel_x]
            numpy_train[filter, yVelocity] = [x for x in smoothed_vel_y]
            numpy_train[filter, xAcceleration] = [x for x in smoothed_accel_x]
            numpy_train[filter, yAcceleration] = [x for x in smoothed_accel_y]


    # print to file
    file_name = file_names
    file_path = path_to_smoothed_dataset + file_name + '_smoothed_' + str(window) + '.csv'
    df = pd.DataFrame(numpy_train, columns=columns)
    df.to_csv(file_path, index=False, header=True)
    # with open(file_path, 'w') as f:
    #     np.savetxt(fname=file_path, X=numpy_train, header=columns, delimiter=",")


def main():
    # smooth window must be an odd value
    smoothing_window = 21
    print(f"Smoothing window is set to {str(smoothing_window)}")

    # change the file names as needed
    global file_names
    file_names = 'i80.csv'
    # file_names = '0750_0805_us101_smoothed_21_.csv'
    # define the index of columns containing vehicle id, time, local x, local y, velocity and acceleration
    # these indexes correspond to the original dataset if not modified
    # the indexes help treat the dataset as matrix and perform smoothing using matrix functions
    global vehicle_id, time_column, local_x, local_y, v_vel, v_acc, xVelocity, yVelocity, xAcceleration, yAcceleration
    vehicle_id, time_column, local_x, local_y, v_vel, v_acc, xVelocity, yVelocity, xAcceleration, yAcceleration  = 0, 3, 4, 5, 11, 12, 18, 19, 20, 21


    # specify the path to the input NGSIM dataset and the path to the output smoothed dataset
    global path_to_dataset, path_to_smoothed_dataset
    path_to_dataset = 'data/raw/'
    path_to_smoothed_dataset = 'data/smooth/'


    # load the NGSIM data from the CSV files
    train = pd.read_csv(path_to_dataset + file_names, engine='c')

    train.insert(loc=18, column='xVelocity', value=0)
    train.insert(loc=19, column='yVelocity', value=0)
    train.insert(loc=20, column='xAcceleration', value=0)
    train.insert(loc=21, column='yAcceleration', value=0)

    global columns
    columns = train.columns.values
    smooth_dataset(smoothing_window, train, file_names)

if __name__ == '__main__':
    main()
