import time
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import matplotlib.pyplot as plt
import copy
from threading import Thread
import pickle
import os
from statistics import mean


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.epoch = 0
        self.dataset = args.dataset
        total_sample = 0
        for client in self.clients:
            total_sample += len(client.train_samples)
        if args.weight1 == 5 and args.weight2 == 5:
            for client in self.clients:
                self.clients_weight.append(len(client.train_samples)/total_sample)
        else:
            self.clients_weight.append(args.weight1)
            self.clients_weight.append(args.weight2)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        earlystopping = False
        self.selected_clients = self.clients
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.send_models()
            next_state_a, _, loss, rmse, earlystopping = self.evaluate()
            for j, client in enumerate(self.selected_clients):
                client.train(self.epoch)
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate personal model")
            next_state_a, _, loss, rmse, earlystopping = self.evaluate()
            self.states_lst.append([0.1 if value > 0.1 else value for value in next_state_a])
            self.receive_models()

            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            self.epoch = i
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            if earlystopping:
                break

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        print("Best metrics:")
        min_value = min(self.val_rmse)
        print("Val RMSE")
        print(min_value)
        print("Epochs")
        min_index = self.val_rmse.index(min_value)
        print(min_index / 2)
        print("Test RMSE")
        print(self.test_rmse[min_index])
        print("Test ADE")
        print(self.test_mae[min_index])
        print("Test FDE")
        print(self.test_mape[min_index])
        print("Best metrics in each client:")
        best_rmse = []
        best_mae = []
        best_mape = []
        weight = []
        for i, client in enumerate(self.clients):
            weight.append(len(client.train_samples))
            min_value = min(client.val_rmse)
            print("Client" + str(i) + "  Val RMSE")
            print(min_value)
            min_index = client.val_rmse.index(min_value)
            print("Epochs")
            print(min_index)
            print("Client " + str(i) + " Test RMSE")
            print(client.test_rmse[min_index])
            best_rmse.append(client.test_rmse[min_index])
            print("Client " + str(i) + "Test ADE")
            print(client.test_mae[min_index])
            best_mae.append(client.test_mae[min_index])
            print("Client " + str(i) + "Test FDE")
            print(client.test_mape[min_index])
            best_mape.append(client.test_mape[min_index])
        print("Best metrics totally:")
        print("Best RMSE")
        print(sum([x * y for x, y in zip(best_rmse, weight)]) / sum(weight))
        print("Best ADE")
        print(sum([x * y for x, y in zip(best_mae, weight)]) / sum(weight))
        print("Best FDE")
        print(sum([x * y for x, y in zip(best_mape, weight)]) / sum(weight))

        for i, client in enumerate(self.clients):  # todo 画图
            fig, bx = plt.subplots()
            print(client.test_rmse)
            bx.plot(client.test_rmse, label='test RMSE')
            bx.set_xlabel('Epoch')
            bx.set_ylabel('RMSE')
            bx.set_title('test RMSE vs Epoch')
            bx.legend()
            self.save_figure(fig, 'client_rmse', i)
        for i, client in enumerate(self.clients):
            fig, bx = plt.subplots()
            print(client.val_loss)
            bx.plot(client.val_loss, label='test loss')
            bx.set_xlabel('Epoch')
            bx.set_ylabel('loss')
            bx.set_title('test loss vs Epoch')
            bx.legend()
            self.save_figure(fig, 'client_loss', i)

        color_box = ['r', 'b', 'g', 'c', 'y', 'k']
        fig, bx = plt.subplots()
        print(self.test_rmse)
        bx.plot(self.test_rmse, label='val RMSE')
        bx.set_xlabel('Epoch')
        bx.set_ylabel('RMSE')
        bx.set_title('val RMSE vs Epoch')
        bx.legend()
        self.save_figure(fig, 'server_rmse', 0)

        fig, ax = plt.subplots()
        for i in range(self.num_clients):
            state1_lst = []
            for row in self.states_lst:
                state1_lst.append(row[i])
            # print(state1_lst)
            ax.plot(state1_lst, color_box[i], label='client' + str(i))
        ax.set_title('loss vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('loss')
        ax.legend()
        self.save_figure(fig, 'loss', -1)

        rmse_lst = []
        for i, client in enumerate(self.clients):
            rmse_lst.append(client.test_rmse)
        fig, ax = plt.subplots()
        for i in range(self.num_clients):
            ax.plot(rmse_lst[i], color_box[i], label='client' + str(i))
        ax.plot(self.val_rmse, 'g', label='server')
        ax.set_title('rmse vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('rmse')
        ax.legend()
        self.save_figure(fig, 'rmse', -1)

        save_data = {
            # 'Average time cost per round': sum(self.Budget[1:])/len(self.Budget[1:]),
            'Best metrics': {
                'Val RMSE': min_value,
                'Epochs': min_index,
                'Test RMSE': self.test_rmse[min_index],
                'Test MAE': self.test_mae[min_index],
                'Test MAPE': self.test_mape[min_index]
            }
        }
        save_data2 = []
        for i, client in enumerate(self.clients):
            tmp = {
                'client.val_rmse': client.val_rmse,
                'client.val_loss': client.val_loss,
                # 'client.train_loss': client.train_loss
            }
            save_data2.append(tmp)
        save_data3 = {
            'val_rmse': self.val_rmse,
        }
        result_path = os.path.join(self.model_path, 'output.pkl')
        # 将数据序列化并写入文件
        with open(result_path, 'wb') as f:
            pickle.dump(save_data, f)
            pickle.dump(save_data2, f)
            pickle.dump(save_data3, f)

        return self.epoch
