import torch
from torch.utils.data import DataLoader
import datetime
import time
import argparse

from flcore.trainmodel.lstm_mine import LSTMNet_trajectory
from flcore.trainmodel.stgcn import social_stgcnn
from utils.metrics import seq_to_nodes, nodes_rel_to_nodes_abs, rmse, ade
from utils.trajectory_utils import TrajectoryDataset

obs_seq_len = 15
pred_seq_len = 25

def run(args):
    model_str = args.model
    if model_str == "stgcn":
        model = social_stgcnn(n_stgcnn=args.n_stgcnn,n_txpcnn=args.n_txpcnn,output_feat=5,seq_len=obs_seq_len,kernel_size=args.kernel_size,pred_seq_len=pred_seq_len).to(args.device)
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load('models/' + args.dataset + 'NGSIM2/model_FedAvg_NGSIM2_2024-02-19_23-29-12/FedAvg_bestmodel.pth'))
    # model = torch.load('models/NGSIM2/model_FedAvg_NGSIM2_2024-02-19_23-29-12/FedAvg_bestmodel.pth')
    model.eval()
    test_rmse = []
    number_of = []

    for i in range(args.num_clients):
        data_set = '../dataset/' + model_str + '/rawdata/'
        dset_test = TrajectoryDataset(
            data_set + 'test/' + str(i) + '/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1, norm_lap_matr=True
            , delim=','
        )
        loader_test = DataLoader(
            dset_test,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=False,
            num_workers=1)
        ade_bigls = []
        fde_bigls = []
        batch_count = 0
        with torch.no_grad():
            for cnt, batch in enumerate(loader_test):
                batch_count += 1
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)
                V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
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
                ade_bigls.append(rmse(pred, target, number_of))
            test_rmse.append(sum(ade_bigls) / len(ade_bigls))
            number_of.append(len(ade_bigls))


    rmse = 0.0
    num = 0.0
    for i in range(len(test_rmse)):
        rmse += test_rmse[i] * number_of[i]
        num += number_of[i]
    rmse = rmse / num
    print("Averaged Test RMSE: {:.4f}".format(rmse))


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
    parser.add_argument('-ks', "--kernel_size", type=int, default=7)
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
    parser.add_argument('-bt', "--beta", type=float, default=0.1,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0.001,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)

    parser.add_argument('-es', "--earlystop", type=int, default=0,
                        help="use earlystopping or not")
    parser.add_argument('-f', "--flag", type=str, default="None",
                        help="introduction")
    parser.add_argument('-stg', "--n_stgcnn", type=int, default=3)
    parser.add_argument('-txp', "--n_txpcnn", type=int, default=5)
    args = parser.parse_args()

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
    print(args.flag)

    if args.device == "cuda":
        print("Cuda device id: {}".format(args.device_id))
    print("=" * 50)

    run(args)



