import torch

dqn_model = torch.load('/data/why/workspace/FederatedLearning/system_trajectory_reinforcement/models/NGSIM/FedAvg_dqnmodel_0.pt')
dqn_model.eval()
state = [0.0055, 0.0032]
# state = [0.0012, 0.0028680218931506664]
with torch.no_grad():
    state = torch.unsqueeze(torch.FloatTensor(state), 0)
    actions_value = dqn_model.forward(state)
action = actions_value[0].tolist()
print("客户端权重：")
print(action)