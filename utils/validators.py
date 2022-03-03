import torch
import loguru
import numpy as np

from loguru import logger
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score


class BaseValidator:
    def __init__(self,
                 valid_loader: DataLoader,
                 device):

        self.valid_loader = valid_loader
        self.device = device

    def model_out(self, model):
        model.eval()

        with torch.no_grad():
            for batch_data in self.valid_loader:
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                predicts, _ = model(**batch_data)

                yield predicts, batch_data['labels']


class DT1Validator(BaseValidator):
    def __init__(self,
                 valid_loader: DataLoader,
                 device):
        super(DT1Validator, self).__init__(valid_loader=valid_loader,
                                           device=device)

    def __call__(self, model):
        predict_list = []
        predict_score_list = []
        label_list = []
        for predicts, labels in self.model_out(model):
            if predicts.shape != labels.shape:
                labels = labels.squeeze(1)
            is_valid = labels.reshape(-1) ** 2 > 0
            predict_list.extend([1 if ele > 0.5 else 0 for ele in predicts.reshape(-1)[is_valid].cpu().numpy()])
            predict_score_list.extend(predicts.reshape(-1)[is_valid].cpu().numpy())
            label_list.extend(((labels.reshape(-1)+1)/2)[is_valid].cpu().numpy())

        return accuracy_score(predict_list, label_list), roc_auc_score(label_list, predict_score_list)

