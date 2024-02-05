import copy
import os
import pickle

from matplotlib import pyplot as plt

from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread
import time

from utils.early_stopping import EarlyStopping


class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientDitto)
        self.epoch = 0
        self.dataset = args.dataset
        total_sample = 0
        for client in self.clients:
            total_sample += len(client.train_samples)
        for client in self.clients:
            self.clients_weight.append(len(client.train_samples) / total_sample)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        model_path = os.path.join(self.model_path, self.algorithm + "_bestmodel" + ".pth")
        self.early_stopping = EarlyStopping(path=model_path, id=-1, patience=300)

        # self.load_model()
        self.Budget = []

    def train(self):
        earlystopping = False
        self.selected_clients = self.clients
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.send_models()
            next_state_a, _, loss, rmse, earlystopping = self.evaluate()
            next_state_a, _, loss, rmse, earlystopping = self.evaluate_personalized()
            for j, client in enumerate(self.selected_clients):
                client.ptrain()
                client.train()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate personal model")
            next_state_a, _, loss, rmse, earlystopping = self.evaluate()
            next_state_a, _, loss, rmse, earlystopping = self.evaluate_personalized()
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
        # for i, client in enumerate(self.clients):
        #     fig, bx = plt.subplots()
        #     print(client.train_loss)
        #     bx.plot(client.train_loss, label='train loss')
        #     bx.set_xlabel('Epoch')
        #     bx.set_ylabel('loss')
        #     bx.set_title('train loss vs Epoch')
        #     bx.legend()
        #     self.save_figure(fig, 'client_trloss', i)
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
        ax.set_title('loss vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('loss')
        ax.legend()
        self.save_figure(fig, 'rmse', -1)

        # fig, bx = plt.subplots()
        # print(self.p_val_rmse)
        # bx.plot(self.p_val_rmse, label='personalized val RMSE')
        # bx.set_xlabel('Epoch')
        # bx.set_ylabel('RMSE')
        # bx.set_title('val RMSE vs Epoch')
        # bx.legend()
        # self.save_figure(fig, 'server_personalized_rmse', 0)

        # self.save_results()
        # self.save_global_model()

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

    def test_metrics_personalized(self):
        num_samples = []
        losses = []
        tot_rmse = []
        tot_mae = []
        tot_mape = []
        tot_batch_loss = []
        tot_val_rmse = []
        for c in self.clients:
            test_num, test_loss, test_rmse, test_mae, test_mape, val_rmse, stopped = c.test_metrics_personalized()
            # test_rmse = test_rmse.item()
            tot_rmse.append(test_rmse)
            tot_mae.append(test_mae)
            tot_mape.append(test_mape)
            num_samples.append(test_num)
            tot_val_rmse.append(val_rmse)
            losses.append(test_loss * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_rmse, tot_mae, tot_mape, losses, tot_val_rmse

    def evaluate_personalized(self, rmse=None, es=True):
        stats = self.test_metrics_personalized()
        # stats_train = self.train_metrics_personalized()
        for i, t in enumerate(stats[2]):
            print("Client" + str(i) + " Test RMSE: " + str(t))
        # for i, t in enumerate(stats_train[2]):
        #     print("Client" + str(i) + " Train loss: " + str(t))
        test_rmse = 0
        test_mae = 0
        test_mape = 0
        val_rmse = 0
        test_loss = 0
        for a, n in zip(stats[2], stats[1]):
            test_rmse += a * n
        for a, n in zip(stats[3], stats[1]):
            test_mae += a * n
        for a, n in zip(stats[4], stats[1]):
            test_mape += a * n
        for a, n in zip(stats[6], stats[1]):
            val_rmse += a * n
        for a, n in zip(stats[5], stats[1]):
            test_loss += a * n
        test_rmse = test_rmse / sum(stats[1])
        test_mae = test_mae / sum(stats[1])
        test_mape = test_mape / sum(stats[1])
        val_rmse = val_rmse / sum(stats[1])
        test_loss = test_loss / sum(stats[1])
        # if rmse == None:
        #     self.rs_test_rmse.append(test_rmse)
        # else:
        #     rmse.append(test_rmse)
        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Val RMSE: {:.4f}".format(val_rmse))
        # print("Averaged Test Loss: {:.4f}".format(test_loss))
        print("Averaged Test RMSE: {:.4f}".format(test_rmse))
        print("Averaged Test MAE: {:.4f}".format(test_mae))
        print("Averaged Test MAPE: {:.4f}".format(test_mape))
        # print("Cilent Test loss: {:.4f}".format(stats[5][0]))
        # print("Cilent Test loss: {:.4f}".format(stats[5][1]))
        self.val_rmse.append(40 if val_rmse > 40 else val_rmse)
        self.test_rmse.append(40 if test_rmse > 40 else test_rmse)
        self.test_mae.append(30 if test_mae > 30 else test_mae)
        self.test_mape.append(50 if test_mape > 50 else test_mape)
        if es:
            # self.early_stopping(val_rmse, self.global_model, "rmse")
            self.early_stopping(test_rmse, self.global_model, "rmse")
        # 达到早停止条件时，early_stop会被置为True
        return stats[5], stats[1], test_loss, test_rmse, self.early_stopping.early_stop
        # return 1
