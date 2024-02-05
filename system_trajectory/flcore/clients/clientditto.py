import pandas as pd
import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from utils.metrics import nodes_rel_to_nodes_abs, rmse, ade, fde
from utils.privacy import *
from utils.trajectory_utils import graph_loss


class clientDitto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.plocal_steps = args.plocal_steps

        self.pmodel = copy.deepcopy(self.model)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.poptimizer = PerturbedGradientDescent(
            self.pmodel.parameters(), lr=self.learning_rate, mu=self.mu)

        # differential privacy
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def ptrain(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.pmodel.train()
        loss_batch = 0
        batch_count = 0
        is_fst_loss = True
        loader_len = len(trainloader)
        turn_point = int(loader_len / self.batch_size) * self.batch_size + loader_len % self.batch_size - 1
        max_local_steps = self.plocal_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            batch_count = 0
            for cnt, batch in enumerate(trainloader):
                batch_count += 1
                # Get data
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                self.poptimizer.zero_grad()
                # Forward
                # V_obs = batch,seq,node,feat
                # V_obs_tmp = batch,feat,seq,node
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)
                V_pred, _ = self.pmodel(V_obs_tmp, A_obs.squeeze())
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
                    if batch_count % self.batch_size == 1:
                        continue
                    loss = loss / self.batch_size
                    is_fst_loss = True
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.poptimizer.step(self.model.parameters(), self.device)
                    loss_batch += loss.item()

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()
        loss_batch = 0
        batch_count = 0
        is_fst_loss = True
        loader_len = len(trainloader)
        turn_point = int(loader_len / self.batch_size) * self.batch_size + loader_len % self.batch_size - 1
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            batch_count = 0
            for cnt, batch in enumerate(trainloader):
                batch_count += 1
                # Get data
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                self.optimizer.zero_grad()
                # Forward
                # V_obs = batch,seq,node,feat
                # V_obs_tmp = batch,feat,seq,node
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
                    if batch_count % self.batch_size == 1:
                        continue
                    loss = loss / self.batch_size
                    is_fst_loss = True
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    # Metrics
                    loss_batch += loss.item()

        self.train_loss.append(loss_batch / batch_count)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")

    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.pmodel.eval()

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
                V_pred, _ = self.pmodel(V_obs_tmp, A_obs.squeeze())
                V_pred = V_pred.permute(0, 2, 3, 1)
                V_tr = V_tr.squeeze()
                A_tr = A_tr.squeeze()
                V_pred = V_pred.squeeze()
                num_of_objs = obs_traj_rel.shape[1]
                V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
                mean = V_pred[:, :, 0:2]
                from utils.metrics import seq_to_nodes
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
                    # val_batch_loss.append(loss.item() / cut)
                    # todo 没做

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

                # batch_rmse, batch_cnt = rmse(pred, target, number_of, epoch)
                # all_rmse += batch_rmse
                # all_cnt += batch_cnt

                if batch_count <= 0.5 * loader_len:
                    rmse_bigls.append(rmse(pred, target, number_of))
                    mae_bigls.append(ade(pred, target, number_of))
                    mape_bigls.append(fde(pred, target, number_of))
                else:
                    test_rmse_bigls.append(rmse(pred, target, number_of))
                    test_mae_bigls.append(ade(pred, target, number_of))
                    test_mape_bigls.append(fde(pred, target, number_of))

                # rmse_vec = rmse(pred, target, number_of)
                # test_rmse_bigls.append(rmse_vec)
                # mae_vec = ade(pred, target, number_of)
                # test_mae_bigls.append(mae_vec)
                # mape_vec = fde(pred, target, number_of)
                # test_mape_bigls.append(mape_vec)
                # print(rmse_vec)
                # if epoch < 150 or rmse_vec < 10:
                #     test_rmse_bigls.append(rmse_vec)
                # test_mae_bigls.append(mae(pred, target, number_of))
                # test_mape_bigls.append(mape(pred, target, number_of))

                # for n in range(num_of_objs):
                #     pred = []
                #     target = []
                #     obsrvs = []
                #     number_of = []
                #     pred.append(V_pred_rel_to_abs[:, n:n + 1, :])#todo 预测图上全部车辆
                #     target.append(V_y_rel_to_abs[:, n:n + 1, :])
                #     obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                #     number_of.append(1)
                #     ade_bigls.append(rmse(pred,target,number_of))
                #     fde_bigls.append(ade(pred,target,number_of))
            rmse_ = sum(rmse_bigls) / len(rmse_bigls)
            mae_ = sum(mae_bigls) / len(mae_bigls)
            mape_ = sum(mape_bigls) / len(mape_bigls)
            test_rmse_ = sum(test_rmse_bigls) / len(test_rmse_bigls)
            test_mae_ = sum(test_mae_bigls) / len(test_mae_bigls)
            test_mape_ = sum(test_mape_bigls) / len(test_mape_bigls)
            # self.val_rmse.append(rmse_)
            # self.val_loss.append(loss_batch / batch_count)#todo 只是最后batch的loss
            # self.test_rmse.append(test_rmse_)
            # self.prev_rmse = self.rmse
            # self.rmse = rmse_

            if not pd.isna(rmse_):
                self.val_rmse.append(40 if rmse_ > 40 else rmse_)
            if not pd.isna(loss_batch / batch_count):
                self.val_loss.append(loss_batch / batch_count)  # todo 只是最后batch的loss
            # test_rmse_ = math.pow(all_rmse / all_cnt, 0.5)*0.3048
            if not pd.isna(test_rmse_):
                self.test_rmse.append(test_rmse_)
            if not pd.isna(test_mae_):
                self.test_mae.append(test_mae_)
            if not pd.isna(test_mape_):
                self.test_mape.append(test_mape_)
        # return batch_count, loss_batch / batch_count, test_rmse_, test_mae_, test_mape_, rmse_, False
        return batch_count, loss_batch / batch_count, test_rmse_, test_mae_, test_mape_, rmse_, False
