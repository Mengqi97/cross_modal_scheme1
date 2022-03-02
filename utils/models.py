import os
import sys

import torch
import torch.nn as nn
from transformers import BertModel
from loguru import logger


base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)


class BaseModel(nn.Module):
    def __init__(self, _config):

        super(BaseModel, self).__init__()
        self.mode = _config.mode
        if _config.bert_dir:
            bert_dir = os.path.join(base_dir, _config.bert_dir)
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            self.bert = BertModel.from_pretrained(_config.bert_name)
        self.bert_config = self.bert.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)


class ClintoxModel(BaseModel):
    def __init__(self,
                 _config,
                 task_num):

        super(ClintoxModel, self).__init__(_config)

        # 扩充词表故需要重定义
        self.bert.pooler = None
        self.bert.resize_token_embeddings(_config.len_of_tokenizer)
        out_dims = self.bert_config.hidden_size
        # mid_linear_dims = _config.mid_linear_dims

        # 下游任务模型结构构建
        # self.mid_linear = nn.Sequential(
        #     nn.Linear(out_dims, mid_linear_dims),
        #     nn.ReLU(),
        #     nn.Dropout(_config.dropout_prob)
        # )
        # self.classifier = nn.Linear(mid_linear_dims, 1)
        self.classifier = nn.Linear(out_dims, task_num)
        self.activation = nn.Sigmoid()
        if 'train' == self.mode:
            self.criterion = nn.BCELoss(reduction='none')
        else:
            self.criterion = None

        # 模型初始化
        init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        out = out.last_hidden_state[:, 0, :]  # 取cls对应的embedding
        # out = self.mid_linear(out)
        out = self.activation(self.classifier(out))

        if self.criterion:
            if out.shape != labels.shape:
                labels = torch.squeeze(labels)

            loss = self.criterion(out, labels)
            # loss = self.criterion(out, (labels+1)/2)

            return out, loss

        return out

