#!/usr/bin/env python
import copy
import datetime

import torch
import argparse
import os
import time
import warnings
import numpy as np
from torch.nn.functional import dropout
import torchvision

from flcore.servers.serveravg_pure import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serveropt import FedOPT
from flcore.servers.serverlocal import Local
from flcore.servers.serverditto import Ditto


from flcore.trainmodel.models import *

from flcore.trainmodel.stgcn import social_stgcnn
# from flcore.trainmodel.resnet import resnet18 as resnet

warnings.simplefilter("ignore")
torch.manual_seed(0)
print("seed 0")

# hyper-params for trajectory tasks
obs_seq_len = 15
pred_seq_len = 25

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    all_size = (param_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    print('模型参数量为：{:.3f}'.format(param_sum))


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model
    epochs = args.global_rounds

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "stgcn":
            args.model = social_stgcnn(n_stgcnn=args.n_stgcnn,n_txpcnn=args.n_txpcnn,output_feat=5,seq_len=obs_seq_len,kernel_size=7,pred_seq_len=pred_seq_len).to(args.device)
            args.modelname = model_str
            getModelSize(args.model)
        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == "FedOPT":
            server = FedOPT(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
            
        else:
            raise NotImplementedError

        epochs = server.train()


        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    # average_data(dataset=args.dataset,
    #             algorithm=args.algorithm,
    #             goal=args.goal,
    #             times=args.times,
    #             length=epochs/args.eval_gap+1) #args.global_rounds = epochs

    # print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=16)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-ls', "--local_steps", type=int, default=3)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    torch.cuda.set_device(int(args.device_id))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    print(datetime.datetime.now())
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("n_stgcnn: {}".format(args.n_stgcnn))
    print("n_txpcnn: {}".format(args.n_txpcnn))
    print("weight1: {}".format(args.weight1))
    print("weight2: {}".format(args.weight2))
    print(args.flag)

    if args.device == "cuda":
        # print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        print("Cuda device id: {}".format(args.device_id))
    print("=" * 50)

    model_path = os.path.join("models", args.dataset)
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"{date_string}")
    model_path = os.path.join(model_path, f"model_{date_string}")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    args.model_path = model_path
    log_path = os.path.join(model_path, "log" + ".txt")
    run(args)

    print("=" * 50)
    print(datetime.datetime.now())
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("n_stgcnn: {}".format(args.n_stgcnn))
    print("n_txpcnn: {}".format(args.n_txpcnn))
    print(args.flag)
