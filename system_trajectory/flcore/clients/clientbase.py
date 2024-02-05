import copy
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.metrics import *
from utils.trajectory_utils import *


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        if args.modelname == 'stgcn':
            self.graph = True
        else:
            self.graph = False

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.sample_rate = self.batch_size / len(self.train_samples)

        self.train_loss = []
        self.val_rmse = []
        self.val_loss = []
        self.test_rmse = []
        self.test_mae = []
        self.test_mape = []
        self.prev_rmse = 0
        self.rmse = 0



    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(self.train_samples, batch_size=1, drop_last=False, shuffle=True, num_workers=0) #原
        # return train_data
    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(self.test_samples, batch_size=1, drop_last=False, shuffle=True, num_workers=0)#todo 顺序加载
        # return test_data
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()
        # self.model.to(self.device)
        test_num = 0
        rmse_bigls = []
        mae_bigls = []
        mape_bigls = []
        test_rmse_bigls = []
        test_mae_bigls = []
        test_mape_bigls = []
        all_rmse = 0
        all_cnt = 0
        batch_count = 0
        loss_batch = 0
        is_fst_loss = True
        loader_len = len(testloaderfull)
        turn_point = int(loader_len / self.batch_size) * self.batch_size + loader_len % self.batch_size - 1
        val_batch_loss = []
        with torch.no_grad():
            for cnt, batch in enumerate(testloaderfull):
                batch_count += 1
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)
                V_pred, _ = self.model(V_obs_tmp, A_obs.squeeze())
                V_pred = V_pred.permute(0, 2, 3, 1)
                V_tr = V_tr.squeeze()
                A_tr = A_tr.squeeze()
                V_pred = V_pred.squeeze()
                num_of_objs = obs_traj_rel.shape[1]
                V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
                mean = V_pred[:, :, 0:2]
                V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
                V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                        V_x[0, :, :].copy())

                V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
                V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                        V_x[-1, :, :].copy())
                if batch_count % self.batch_size != 0 and cnt != turn_point:
                    l = graph_loss(V_pred, V_tr)
                    if is_fst_loss:
                        loss = l
                        is_fst_loss = False
                    else:
                        loss += l
                else:
                    if cnt == turn_point:
                        cut = loader_len % self.batch_size
                    else:
                        cut = self.batch_size
                    loss = loss / self.batch_size
                    is_fst_loss = True
                    # Metrics
                    loss_batch += loss.item()


                V_pred = mean
                V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                            V_x[-1, :, :].copy())
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, 0:1, :])  # todo 只预测中央车辆
                target.append(V_y_rel_to_abs[:, 0:1, :])
                obsrvs.append(V_x_rel_to_abs[:, 0:1, :])
                number_of.append(1)
                if batch_count <= 0.5 * loader_len:
                    rmse_bigls.append(rmse(pred, target, number_of))
                    mae_bigls.append(ade(pred, target, number_of))
                    mape_bigls.append(fde(pred, target, number_of))
                else:
                    test_rmse_bigls.append(rmse(pred, target, number_of))
                    test_mae_bigls.append(ade(pred, target, number_of))
                    test_mape_bigls.append(fde(pred, target, number_of))
            rmse_ = sum(rmse_bigls) / len(rmse_bigls)
            mae_ = sum(mae_bigls) / len(mae_bigls)
            mape_ = sum(mape_bigls) / len(mape_bigls)
            test_rmse_ = sum(test_rmse_bigls) / len(test_rmse_bigls)
            test_mae_ = sum(test_mae_bigls) / len(test_mae_bigls)
            test_mape_ = sum(test_mape_bigls) / len(test_mape_bigls)

            if not pd.isna(rmse_):
                self.val_rmse.append(40 if rmse_ > 40 else rmse_)
            if not pd.isna(loss_batch / batch_count):
                self.val_loss.append(loss_batch / batch_count)#todo 只是最后batch的loss
            if not pd.isna(test_rmse_):
                self.test_rmse.append(test_rmse_)
            if not pd.isna(test_mae_):
                self.test_mae.append(test_mae_)
            if not pd.isna(test_mape_):
                self.test_mape.append(test_mape_)
            return batch_count, loss_batch / batch_count, test_rmse_, test_mae_, test_mape_, rmse_, False



    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        self.model.eval()
        rmse_bigls = []
        mae_bigls = []
        mape_bigls = []
        test_rmse_bigls = []
        test_mae_bigls = []
        test_mape_bigls = []
        batch_count = 0
        loss_batch = 0
        is_fst_loss = True
        loader_len = len(testloaderfull)
        turn_point = int(loader_len / self.batch_size) * self.batch_size + loader_len % self.batch_size - 1
        with torch.no_grad():
            for cnt, batch in enumerate(testloaderfull):
                batch_count += 1
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)
                V_pred, _ = self.model(V_obs_tmp, A_obs.squeeze())
                V_pred = V_pred.permute(0, 2, 3, 1)
                V_tr = V_tr.squeeze()
                A_tr = A_tr.squeeze()
                V_pred = V_pred.squeeze()
                num_of_objs = obs_traj_rel.shape[1]
                V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
                mean = V_pred[:, :, 0:2]
                V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
                V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                        V_x[0, :, :].copy())

                V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
                V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                        V_x[-1, :, :].copy())
                if batch_count % self.batch_size != 0 and cnt != turn_point:
                    l = graph_loss(V_pred, V_tr)
                    if is_fst_loss:
                        loss = l
                        is_fst_loss = False
                    else:
                        loss += l
                else:
                    loss = loss / self.batch_size
                    is_fst_loss = True
                    # Metrics
                    loss_batch += loss.item()

                V_pred = mean
                V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                           V_x[-1, :, :].copy())
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, 0:1, :])  # 只预测中央车辆
                target.append(V_y_rel_to_abs[:, 0:1, :])
                obsrvs.append(V_x_rel_to_abs[:, 0:1, :])
                number_of.append(1)
                if batch_count <= 0.5 * loader_len:
                    rmse_bigls.append(rmse(pred, target, number_of))
                    mae_bigls.append(mae(pred, target, number_of))
                    mape_bigls.append(mape(pred, target, number_of))
                else:
                    test_rmse_bigls.append(rmse(pred, target, number_of))
                    test_mae_bigls.append(mae(pred, target, number_of))
                    test_mape_bigls.append(mape(pred, target, number_of))
            rmse_ = sum(rmse_bigls) / len(rmse_bigls)
            mae_ = sum(mae_bigls) / len(mae_bigls)
            mape_ = sum(mape_bigls) / len(mape_bigls)
            test_rmse_ = sum(test_rmse_bigls) / len(test_rmse_bigls)
            test_mae_ = sum(test_mae_bigls) / len(test_mae_bigls)
            test_mape_ = sum(test_mape_bigls) / len(test_mape_bigls)
        return batch_count, loss_batch / batch_count, test_rmse_, test_mae_, test_mape_, rmse_, False

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        batch_count = 0
        loss_batch = 0
        is_fst_loss = True
        loader_len = len(trainloader)
        turn_point = int(loader_len / self.batch_size) * self.batch_size + loader_len % self.batch_size - 1

        for cnt, batch in enumerate(trainloader):
            batch_count += 1
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch
            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = self.model(V_obs_tmp, A_obs.squeeze())
            V_pred = V_pred.permute(0, 2, 3, 1)
            V_tr = V_tr.squeeze()
            A_tr = A_tr.squeeze()
            V_pred = V_pred.squeeze()

            if batch_count % self.batch_size != 0 and cnt != turn_point:
                l = graph_loss(V_pred, V_tr)
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l
            else:
                loss = loss / self.batch_size
                is_fst_loss = True
                # Metrics
                loss_batch += loss.item()

        return batch_count, loss_batch/batch_count


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

