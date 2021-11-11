import torch
import loguru
import numpy as np

from loguru import logger
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


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


class ClintoxValidator(BaseValidator):
    def __init__(self,
                 valid_loader: DataLoader,
                 device):
        super(ClintoxValidator, self).__init__(valid_loader=valid_loader,
                                               device=device)

    def __call__(self, model):
        predicts_list = []
        labels_list = []
        for predicts, labels in self.model_out(model):
            predicts_list.extend([1 if ele[0] > 0 else 0 for ele in predicts.cpu().numpy()])
            labels_list.extend([1 if ele[0] > 0 else 0 for ele in labels.cpu().numpy()])
        return accuracy_score(predicts_list, labels_list)

