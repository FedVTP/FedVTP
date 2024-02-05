import numpy as np
import pandas as pd
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=200, verbose=True, delta=0, path='/', trace_func=print, id=-1):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.device = ""
        if id == -1:
            self.device = "server"
        else:
            self.device = "client" + str(id)


    def __call__(self, val, model=None, monitor="loss"):

        if monitor == "loss" or monitor == "rmse":
            score = -val
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val, model, monitor)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'{self.device} EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                if pd.isna(score):
                    self.early_stop = True
            elif pd.isna(score):
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val, model, monitor)
                self.counter = 0
        else:
            score = val
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val, model, monitor)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'{self.device}EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            elif pd.isna(score):
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val, model, monitor)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, monitor):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation {monitor} improved ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.path != None:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


        # model_path = os.path.join("models", self.dataset)
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        # torch.save(self.global_model, model_path)