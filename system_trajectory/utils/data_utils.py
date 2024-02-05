import ujson
import numpy as np
import os
import torch
from utils.trajectory_utils import *

# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# NUM_CHANNELS = 1

# IMAGE_SIZE_CIFAR = 32
# NUM_CHANNELS_CIFAR = 3


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    # np.random.seed(100)
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS" :
        return read_client_data_text(dataset, idx)
    if dataset[:2] == "NG" or dataset[:2] == "HI" or dataset[:2] == "ME" or dataset[:2] == "CI" or dataset[:2] == "UT":
        return read_client_data_trajectory(dataset, idx, is_train)


    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(y_train).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data

def read_client_data_timeseries(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

def read_client_data_trajectory(dataset, idx, is_train=True):
    # if is_train:
    #     train_data = read_data(dataset, idx, is_train)
    #     obs_traj_ = torch.Tensor(train_data['obs_traj']).type(torch.float32)
    #     pred_traj_ = torch.Tensor(train_data['pred_traj']).type(torch.float32)
    #     train_data = [(obs_traj, pred_traj) for obs_traj, pred_traj in zip(obs_traj_, pred_traj_)]
    #     return train_data
    # else:
    #     test_data = read_data(dataset, idx, is_train)
    #     obs_traj_ = torch.Tensor(test_data['obs_traj']).type(torch.float32)
    #     pred_traj_ = torch.Tensor(test_data['pred_traj']).type(torch.float32)
    #     test_data = [(obs_traj, pred_traj) for obs_traj, pred_traj in zip(obs_traj_, pred_traj_)]
    #     return test_data

    obs_seq_len = 15
    pred_seq_len = 25
    if dataset == 'HIGHDB':
        data_set = '../dataset/HIGHD/rawdata/'
    else:
        data_set = '../dataset/' + dataset + '/rawdata/'
    if is_train:
        dset_train = TrajectoryDataset(
            data_set + 'train/' + str(idx) + '/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1, norm_lap_matr=True
            , delim=',', id=idx, dataset=dataset, is_train=True
        )
        return dset_train
    else:
        dset_val = TrajectoryDataset(
            data_set + 'val/' + str(idx) + '/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1, norm_lap_matr=True
            , delim=',', id=idx, dataset=dataset, is_train=False
        )
        return dset_val