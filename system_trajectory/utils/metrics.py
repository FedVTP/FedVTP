import math
import numpy as np
import torch


def rmse(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    counts = 0
    lossVal = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
    #     pred = torch.from_numpy(np.swapaxes(predAll[s][:, :count_[s], :], 0, 1))
    #     target = torch.from_numpy(np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1))
    #     muX = pred[:, :, 0]
    #     muY = pred[:, :, 1]
    #     x = target[:, :, 0]
    #     y = target[:, :, 1]
    #     out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    #     lossVal = torch.sum(out, dim=1)
    #     counts = torch.numel(out)
    # return lossVal, counts
        N = pred.shape[0]
        T = pred.shape[1]
        cnt = N * T
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_vec = (pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2
                sum_ += sum_vec
        sum_all += sum_ / (cnt)
        lossVal += sum_
        counts += N * T
    # return lossVal, counts
    return math.sqrt(sum_all / All) * 0.3048  # Calculate RMSE and convert from feet to meters

def old_rmse(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += (pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2
        sum_all += sum_ / (N * T)

    return math.sqrt(sum_all / All) * 0.3048  # Calculate RMSE and convert from feet to meters

def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    counts = 0
    lossVal = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
    #     pred = torch.from_numpy(np.swapaxes(predAll[s][:, :count_[s], :], 0, 1))
    #     target = torch.from_numpy(np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1))
    #     muX = pred[:, :, 0]
    #     muY = pred[:, :, 1]
    #     x = target[:, :, 0]
    #     y = target[:, :, 1]
    #     out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    #     lossVal = torch.sum(out, dim=1)
    #     counts = torch.numel(out)
    # return lossVal, counts
        N = pred.shape[0]
        T = pred.shape[1]
        cnt = N * T
        sum_ = 0
        for i in range(N):
            for t in range(T-5, T):
                sum_vec = (pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2
                sum_ += sum_vec
        sum_all += sum_ / (cnt)
        lossVal += sum_
        counts += N * T
    # return lossVal, counts
    return math.sqrt(sum_all / All) * 0.3048  # Calculate RMSE and convert from feet to meters

def mae(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += np.abs(pred[i, t, 0] - target[i, t, 0]) + np.abs(pred[i, t, 1] - target[i, t, 1])
        sum_all += sum_ / (N * T)

    return sum_all / All * 0.3048

def mape(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += np.abs((pred[i, t, 0] - target[i, t, 0]) / target[i, t, 0]) + np.abs((pred[i, t, 1] - target[i, t, 1]) / target[i, t, 1])
        sum_all += sum_ / (N * T)

    return sum_all / All * 100

def mhd(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        mhd_sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_aB = []
                for s in range(T):
                    sum_aB.append(
                        math.sqrt((pred[i, t, 0] - target[i, s, 0]) ** 2 + (pred[i, t, 1] - target[i, s, 1]) ** 2))
                mhd_sum_ += min(sum_aB)
            mhd_sum_ /= T
        sum_all += mhd_sum_

    return (sum_all / All) * 0.3048  # convert from feet to meters

def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return (sum_all / All) * 0.3048  # convert from feet to meters


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N)

    return (sum_all / All) * 0.3048  # convert from feet to meters

def nll_loss(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0

    for s in range(All):
        pred = torch.from_numpy(np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)).float()
        target = torch.from_numpy(np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)).float()
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sigma = torch.ones(2) * 0.1  # 预测方差
                mu = pred[i, t, :]
                x = target[i, t, :]
                norm = torch.distributions.Normal(mu, sigma)
                sum_ += -norm.log_prob(x).sum()
        sum_all += sum_ / (N * T)

    return sum_all / All

def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]  # number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]

    return nodes_.squeeze()


def closer_to_zero(current, new_v):
    dec = min([(abs(current), current), (abs(new_v), new_v)])[1]
    if dec != current:
        return True
    else:
        return False


def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result

def rmse_loss(V_pred, V_trgt):
    pred = V_pred[:, 0:1, :]
    target = V_trgt[:, 0:1, :]
    mse_loss = torch.nn.MSELoss()
    return math.sqrt(mse_loss(pred, target).item())

def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out

    #for i in range(5):
    #    mask[i,:,:] *= (i+1)

    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask) # although both uses out, the average will be correct, 2*out/2
    return lossVal

def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

def horiz_eval(loss_total, n_horiz):
    loss_total = loss_total.cpu().numpy()
    avg_res = np.zeros(n_horiz)
    n_all = loss_total.shape[0]
    n_frames = n_all//n_horiz
    for i in range(n_horiz):
        if i == 0:
            st_id = 0
        else:
            st_id = n_frames*i
        if i == n_horiz-1:
            en_id = n_all-1
        else:
            en_id = n_frames*i + n_frames - 1
        avg_res[i] = np.mean(loss_total[st_id:en_id+1])
    return avg_res
