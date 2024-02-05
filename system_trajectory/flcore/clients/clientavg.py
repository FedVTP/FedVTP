import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from utils.trajectory_utils import *


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=0.0001)  # todo 换优化器
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        # print("used steplr 0.9")

        # differential privacy
        # self.bad_batch_lst = []
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def train(self, epoch=0):
        if self.graph:
            trainloader = self.load_train_data()
        else:
            trainloader = self.train_samples

        start_time = time.time()

        # self.model.to(self.device)

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

    def find_numbers_above_average(self, numbers):
        average = sum(numbers) / len(numbers)
        threshold = average * 1.5  # 平均数的50%阈值
        outlier_indices = []
        for i, num in enumerate(numbers):
            if num > threshold:
                outlier_indices.append(i)
        return outlier_indices
