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

    def add(self, other):
        self.num_seq += len(other)
        self.seq_start_end += other.seq_start_end
        self.obs_traj = torch.cat((self.obs_traj, other.obs_traj), 0)
        self.pred_traj = torch.cat((self.pred_traj, other.pred_traj), 0)
        self.obs_traj_rel = torch.cat((self.obs_traj_rel, other.obs_traj_rel), 0)
        self.pred_traj_rel = torch.cat((self.pred_traj_rel, other.pred_traj_rel), 0)
        self.non_linear_ped = torch.cat((self.non_linear_ped, other.non_linear_ped), 0)
        self.v_obs += other.v_obs
        self.A_obs += other.A_obs
        self.v_pred += other.v_pred
        self.A_pred += other.A_pred

    def findLaneChange(self, id_car_data):
        change_id_lst = []
        lst_Lane_ID = id_car_data[:, 4]
        new_lst_Lane_ID = list(np.diff(lst_Lane_ID))
        lst_Lane_ID = list(lst_Lane_ID)
        if new_lst_Lane_ID.count(-1) > 0 or new_lst_Lane_ID.count(1) > 0:
            for idx in range(len(lst_Lane_ID) - 1):
                if lst_Lane_ID[idx] != lst_Lane_ID[idx + 1]:
                # if lst_Lane_ID[idx] > lst_Lane_ID[idx + 1]:
                    change_id_lst.append(idx)
            return change_id_lst
        else:
            return [-1]

    def getNearLaneCar(self, lane_change_frame_data, data):
        target_frame = lane_change_frame_data[0]
        target_y = lane_change_frame_data[3]
        target_lane = lane_change_frame_data[4]
        this_time_data = data[target_frame == data[:, 0], :]
        near_car_num = 0
        if self.dataset == "NGSIM" or self.dataset == "NGSIM6S" or self.dataset == "NGSIM6N":
            left_lane_box = [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6],
                             [2, 3, 4, 5, 6]]
            right_lane_box = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                              [1, 2, 3, 4, 5]]
        elif self.dataset == "NGSIM2" or self.dataset == "NGSIMO" or self.dataset == "NGSIMS" or self.dataset == "NGSIM6IN2"  or self.dataset == "NGSIM2N":
            left_lane_box = [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]
            right_lane_box = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        elif self.dataset == "NGSIM6IN1" or self.dataset == "NGSIM2IN1":
            left_lane_box = [[2, 3, 4, 5, 6]]
            right_lane_box = [[1, 2, 3, 4, 5]]
        elif self.dataset == "HIGHD" or self.dataset == "HIGHDA":
            left_lane_box = [[3, 4], [3], [3, 4], [3, 4], [3], [3, 4]]
            right_lane_box = [[2, 3], [2], [2, 3], [2, 3], [2], [2, 3, 4]]
        elif self.dataset == "HIGHD1":
            left_lane_box = [[3, 4], [3, 4], [3, 4], [3, 4], [3, 4]]
            right_lane_box = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
        elif self.dataset == "HIGHD6IN1":
            left_lane_box = [[3, 4], [3], [3, 4], [3, 4], [3], [3, 4]]
            right_lane_box = [[2, 3], [2], [2, 3], [2, 3], [2], [2, 3, 4]]
        elif self.dataset == "HIGHDB":
            left_lane_box = [[7, 8], [6], [7, 8], [7, 8], [6], [8, 9]]
            right_lane_box = [[6, 7], [5], [6, 7], [6, 7], [5], [7, 8]]
        elif self.dataset == "MERGE":
            left_lane_box = [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [7, 8], [7, 8], [6], [8, 9], [7, 8], [6]]
            right_lane_box = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [6, 7], [6, 7], [5], [7, 8], [6, 7], [5]]
        elif self.dataset == "UTE":
            left_lane_box = [[7, 8, 9, 10], [6, 7, 8, 9], [7, 8, 9, 10], [2, 3, 4, 5]]
            right_lane_box = [[6, 7, 8, 9], [5, 6, 7, 8], [6, 7, 8, 9], [1, 2, 3, 4]]
        # elif self.dataset == "CITYS":
        #     left_lane_box = [[4, 5]]
        #     right_lane_box = [[3, 4]]
        elif self.dataset == "CITYS3" or self.dataset == "CITYS6IN1":
            left_lane_box = [[4, 5], [1, 2], [2, 3]]
            right_lane_box = [[3, 4], [0, 1], [1, 2]]
            # left_lane_box = [[7, 8], [4, 5], [5, 6]]
            # right_lane_box = [[6, 7], [3, 4], [4, 5]]
        # elif self.dataset == "CITYS3":
        #     left_lane_box = [[4, 5], [5, 6]]
        #     right_lane_box = [[3, 4], [4, 5]]
        elif self.dataset == "CITYS":
            left_lane_box = [[4, 5], [4, 5], [1, 2], [1, 2], [2, 3], [2, 3]]
            right_lane_box = [[3, 4], [3, 4], [0, 1], [0, 1], [1, 2], [1, 2]]
            # left_lane_box = [[7, 8], [7, 8], [4, 5], [4, 5], [5, 6], [5, 6]]
            # right_lane_box = [[6, 7], [6, 7], [3, 4], [3, 4], [4, 5], [4, 5]]
        #左侧车辆
        # if target_lane != 1 and target_lane != 7 and target_lane != 8:
        # if target_lane in left_lane_box[self.id]:
        if self.dataset in ["HIGHDJ", "HIGHDJB"] or (self.dataset in ["HIGHD", "HIGHDB", "HIGHD1", "HIGHD6IN1", "NGSIM", "NGSIMS", "NGSIM2", "MERGE", "UTE", "CITYS3", "CITYS", "NGSIM6IN1", "CITYS6IN1", "NGSIM2IN1", "NGSIM6IN2", "NGSIM2N", "NGSIM6N", "HIGHDA"
            ] and target_lane in left_lane_box[self.id]):
            left_lane_car_data = this_time_data[target_lane-1 == this_time_data[:, 4], :]
            left_lane_car_y = left_lane_car_data[:, 3]
            left_preceding_distance = 9999
            left_following_distance = -9999
            left_preceding_idx = -1
            left_following_idx = -1
            if len(left_lane_car_data) > 0:
                for idx in range(len(left_lane_car_data)):
                    distance = left_lane_car_y[idx] - target_y
                    if distance > 0 and distance < left_preceding_distance:
                        left_preceding_distance = distance
                        left_preceding_idx = idx
                    elif distance < 0 and distance > left_following_distance:
                        left_following_distance = distance
                        left_following_idx = idx
            if left_preceding_distance == 9999:
                left_proceeding_car_data = None
            else:
                left_proceeding_car_id = left_lane_car_data[left_preceding_idx, 1]
                left_preceding_car_alldata = data[left_proceeding_car_id == data[:, 1], :]
                change_idx = -1
                for i in range(len(left_preceding_car_alldata)):
                    if left_preceding_car_alldata[i, 0] == target_frame:
                        change_idx = i
                        break
                if change_idx + self.pred_len <= len(left_preceding_car_alldata) and change_idx - self.obs_len >= 0:
                    left_proceeding_car_data = left_preceding_car_alldata[change_idx-self.obs_len:change_idx+self.pred_len]
                    if self.dataset == "UTE":
                        left_proceeding_car_data[:, 2] = left_proceeding_car_data[:, 2] - self.lane_width
                    near_car_num += 1
                else:
                    # print("临车未全程出现，丢弃")
                    return None, 0
            if left_following_distance == -9999:
                left_following_car_data = None
            else:
                left_following_car_id = left_lane_car_data[left_following_idx, 1]
                left_following_car_alldata = data[left_following_car_id == data[:, 1], :]
                change_idx = -1
                for i in range(len(left_following_car_alldata)):
                    if left_following_car_alldata[i, 0] == target_frame:
                        change_idx = i
                        break
                if change_idx + self.pred_len <= len(left_following_car_alldata) and change_idx - self.obs_len >= 0:
                    left_following_car_data = left_following_car_alldata[
                                               change_idx - self.obs_len:change_idx + self.pred_len]
                    if self.dataset == "UTE":
                        left_proceeding_car_data[:, 2] = left_proceeding_car_data[:, 2] - self.lane_width
                    near_car_num += 1
                else:
                    # print("临车未全程出现，丢弃")
                    return None, 0
        else:
            left_following_car_data = None
            left_proceeding_car_data = None
        #右侧车辆
        # if target_lane != 6 and target_lane != 7 and target_lane != 8:
        # if target_lane in right_lane_box[self.id]:
        if self.dataset in ["HIGHDJ", "HIGHDJB"] or (self.dataset in ["HIGHD", "HIGHDB", "HIGHD1", "HIGHD6IN1", "NGSIM", "NGSIMS", "NGSIM2", "MERGE", "UTE", "CITYS3", "CITYS", "NGSIM6IN1", "CITYS6IN1", "NGSIM2IN1", "NGSIM6IN2", "NGSIM2N", "NGSIM6N", "HIGHDA"
         ] and target_lane in right_lane_box[self.id]):
            right_lane_car_data = this_time_data[target_lane+1 == this_time_data[:, 4], :]
            right_lane_car_y = right_lane_car_data[:, 3]
            right_preceding_distance = 9999
            right_following_distance = -9999
            right_preceding_idx = -1
            right_following_idx = -1
            if len(right_lane_car_data) > 0:
                for idx in range(len(right_lane_car_data)):
                    distance = right_lane_car_y[idx] - target_y
                    if distance > 0 and distance < right_preceding_distance:
                        right_preceding_distance = distance
                        right_preceding_idx = idx
                    elif distance < 0 and distance > right_following_distance:
                        right_following_distance = distance
                        right_following_idx = idx

            if right_preceding_distance == 9999:
                right_proceeding_car_data = None
            else:
                right_proceeding_car_id = right_lane_car_data[right_preceding_idx, 1]
                right_preceding_car_alldata = data[right_proceeding_car_id == data[:, 1], :]
                change_idx = -1
                for i in range(len(right_preceding_car_alldata)):
                    if right_preceding_car_alldata[i, 0] == target_frame:
                        change_idx = i
                        break
                if change_idx + self.pred_len <= len(right_preceding_car_alldata) and change_idx - self.obs_len >= 0:
                    right_proceeding_car_data = right_preceding_car_alldata[
                                               change_idx - self.obs_len:change_idx + self.pred_len]
                    if self.dataset == "UTE":
                        left_proceeding_car_data[:, 2] = left_proceeding_car_data[:, 2] + self.lane_width
                    near_car_num += 1
                else:
                    # print("临车未全程出现，丢弃")
                    return None, 0
            if right_following_distance == -9999:
                right_following_car_data = None
            else:
                right_following_car_id = right_lane_car_data[right_following_idx, 1]
                right_following_car_alldata = data[right_following_car_id == data[:, 1], :]
                change_idx = -1
                for i in range(len(right_following_car_alldata)):
                    if right_following_car_alldata[i, 0] == target_frame:
                        change_idx = i
                        break
                if change_idx + self.pred_len <= len(right_following_car_alldata) and change_idx - self.obs_len >= 0:
                    right_following_car_data = right_following_car_alldata[
                                              change_idx - self.obs_len:change_idx + self.pred_len]
                    if self.dataset == "UTE":
                        left_proceeding_car_data[:, 2] = left_proceeding_car_data[:, 2] + self.lane_width
                    near_car_num += 1
                else:
                    # print("临车未全程出现，丢弃")
                    return None, 0
        else:
            right_following_car_data = None
            right_proceeding_car_data = None
        # 当前车道车辆
        this_lane_car_data = this_time_data[target_lane == this_time_data[:, 4], :]
        this_lane_car_y = this_lane_car_data[:, 3]
        this_preceding_distance = 9999
        this_following_distance = -9999
        this_preceding_idx = -1
        this_following_idx = -1
        if len(this_lane_car_data) > 0:
            for idx in range(len(this_lane_car_data)):
                distance = this_lane_car_y[idx] - target_y
                if distance > 0 and distance < this_preceding_distance:
                    this_preceding_distance = distance
                    this_preceding_idx = idx
                elif distance < 0 and distance > this_following_distance:
                    this_following_distance = distance
                    this_following_idx = idx

        if this_preceding_distance == 9999:
            this_proceeding_car_data = None
        else:
            this_proceeding_car_id = this_lane_car_data[this_preceding_idx, 1]
            this_preceding_car_alldata = data[this_proceeding_car_id == data[:, 1], :]
            change_idx = -1
            for i in range(len(this_preceding_car_alldata)):
                if this_preceding_car_alldata[i, 0] == target_frame:
                    change_idx = i
                    break
            if change_idx + self.pred_len <= len(this_preceding_car_alldata) and change_idx - self.obs_len >= 0:
                this_proceeding_car_data = this_preceding_car_alldata[
                                           change_idx - self.obs_len:change_idx + self.pred_len]
                near_car_num += 1
            else:
                # print("临车未全程出现，丢弃")
                return None, 0
        if this_following_distance == -9999:
            this_following_car_data = None
        else:
            this_following_car_id = this_lane_car_data[this_following_idx, 1]
            this_following_car_alldata = data[this_following_car_id == data[:, 1], :]
            change_idx = -1
            for i in range(len(this_following_car_alldata)):
                if this_following_car_alldata[i, 0] == target_frame:
                    change_idx = i
                    break
            if change_idx + self.pred_len <= len(this_following_car_alldata) and change_idx - self.obs_len >= 0:
                this_following_car_data = this_following_car_alldata[
                                          change_idx - self.obs_len:change_idx + self.pred_len]
                near_car_num += 1
            else:
                # print("临车未全程出现，丢弃")
                return None, 0

        return [left_following_car_data, left_proceeding_car_data, right_following_car_data, right_proceeding_car_data, this_following_car_data, this_proceeding_car_data], near_car_num

    def add_noise_to_data(self, data_list, seed=0,  p=0.5, mean=0, std=1):
        random.seed(seed)
        random_number = random.random()
        if random_number <= p:
            noisy_data_list = []
            for data in data_list:
                if data is None:
                    continue
                noisy_data = data.copy()
                noisy_data[:, 2:4] = self.add_gaussian_noise(noisy_data[:, 2:4], mean, std, seed)
                noisy_data_list.append(noisy_data)
            return noisy_data_list
        else:
            return data_list

    def add_gaussian_noise(self, data, mean=0, std=1, seed=0):
        np.random.seed(seed)
        noise = np.random.normal(mean, std, size=data.shape)
        return data + noise

    def middle_elements(self, my_list):
        l = len(my_list)
        if l < 5:
            return my_list
        mid_index = l // 2
        # 取最中间的两个元素
        middle = my_list[mid_index - 1:mid_index + 2]
        return middle