import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, length=800):
    test_rmse = get_all_results_for_one_algo(
        algorithm, dataset, goal, times, int(length))
    test_acc_data = np.average(test_rmse, axis=0)


    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_rmse[i].max())

    print("std for best rmse:", np.std(max_accurancy))
    print("mean for best rmse:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, length=800):
    test_rmse = np.zeros((times, length))
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + \
            algorithms_list[i] + "_" + goal + "_" + str(i)
        test_rmse[i, :] = np.array(
            read_data_then_delete(file_name, delete=False))[:]

    return test_rmse


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_rmse = np.array(hf.get('rs_test_rmse'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_rmse))

    return rs_test_rmse