import os
import random
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
from utils.metrics import *


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True, id=0, dataset="NGSIM", is_train=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.id = id
        self.dataset = dataset
        self.lane_width = 7
        self.noise = False
        self.seed = 0

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        borderline = 4
        if self.dataset == "NGSIM2N" or self.dataset == "NGSIM6N":
            borderline = 1
        print(str(borderline))
        for path in all_files:
            if self.dataset[:2] == "HI":
                if "site1" in path:
                    self.id = 0
                if "site2" in path:
                    self.id = 1
                if "site3" in path:
                    self.id = 2
                if "site4" in path:
                    self.id = 3
                if "site5" in path:
                    self.id = 4
                if "site6" in path:
                    self.id = 5
            data = read_file(path, delim)
            print("读取了文件" + path)
            cars = np.unique(data[:, 1]).tolist()
            sorted(cars)
            # print(len(cars))
            # print(len(seq_list))
            if self.dataset == "NGSIM2N" or self.dataset == "NGSIM6N" or self.dataset == "HIGHDN":
                train_test_split = round(len(cars)*0.7)
                if is_train:
                    cars = cars[:train_test_split]
                else :
                    cars = cars[train_test_split:]
            car_data = []
            for car in cars:
                car_data.append(data[car == data[:, 1], :])
            for id_car_data in car_data:
                if self.dataset == "HIGHDJ" and id_car_data[0, 3] - id_car_data[1, 3] < 0:
                    continue
                if self.dataset == "HIGHDJB" and id_car_data[0, 3] - id_car_data[1, 3] > 0:
                    continue

                num_frames = len(id_car_data)
                mid_frame = int(num_frames / 2)
                change_id_lst = []
                if num_frames < self.seq_len:
                    continue
                if self.dataset[:2] == "HI":
                    change_id_lst = list(range(self.obs_len, len(id_car_data) - self.pred_len, 10))
                    change_id_lst = self.middle_elements(change_id_lst)
                else:
                    change_id_lst = list(range(self.obs_len, len(id_car_data)-self.pred_len, 60))
                    change_id_lst = self.middle_elements(change_id_lst)
                for change_id in change_id_lst:
                    curr_seq_data = id_car_data[change_id - self.obs_len: change_id + self.pred_len]
                    near_car_list, near_car_num = self.getNearLaneCar(id_car_data[change_id, :], data)
                    if near_car_list is None:
                        # print(str(id_car_data[0, 1]) + "邻车未全程出现，丢弃")
                        continue
                    elif near_car_num < borderline:
                        # print(str(id_car_data[0, 1]) + "邻车少于4辆，丢弃")
                        continue

                    near_car_list.insert(0, curr_seq_data)
                    curr_seq_rel = np.zeros((len(near_car_list), 2, self.seq_len))
                    curr_seq = np.zeros((len(near_car_list), 2, self.seq_len))
                    curr_loss_mask = np.zeros((len(near_car_list), self.seq_len))
                    num_peds_considered = 0
                    _non_linear_ped = []

                    for near_car_data in near_car_list:
                        if near_car_data is None:
                            continue
                        curr_car_seq = np.around(near_car_data, decimals=4)
                        curr_car_seq = np.transpose(curr_car_seq[:, 2:4])
                        rel_curr_car_seq = np.zeros(curr_car_seq.shape)
                        rel_curr_car_seq[:, 1:] = curr_car_seq[:, 1:] - curr_car_seq[:, :-1]
                        _idx = num_peds_considered
                        curr_seq[_idx, :, :] = curr_car_seq
                        curr_seq_rel[_idx, :, :] = rel_curr_car_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(poly_fit(curr_car_seq, pred_len, threshold))
                        curr_loss_mask[_idx, :] = 1
                        num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    # break

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out
