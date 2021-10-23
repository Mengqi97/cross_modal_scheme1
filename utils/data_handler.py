import os
import sys
import random

from config import Config
from utils.functions import SMILES_SPE_Tokenizer

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
from icecream import ic



base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)


class BaseUp:
    def __init__(self,
                 data_path,
                 _config: Config):

        self.max_seq_len = _config.max_seq_len
        self.mode = _config.mode
        self.data = pd.read_csv(data_path)

        if 'default' == _config.tokenizer_txt_type:
            if _config.bert_dir:
                self.tokenizer_txt = BertTokenizer.from_pretrained(os.path.join(base_dir, _config.bert_dir))
            else:
                self.tokenizer_txt = BertTokenizer.from_pretrained(_config.bert_name)
            self.tokenizer_txt.add_tokens(['[SMI]'])
        if 'default' == _config.tokenizer_smi_type:
            self.tokenizer_smi = SMILES_SPE_Tokenizer(
                vocab_file=os.path.join(base_dir, _config.data_dir, _config.spe_voc_file),
                spe_file=os.path.join(base_dir, _config.data_dir, _config.spe_file))


class MLMUp(BaseUp):
    def __init__(self,
                 data_path,
                 _config: Config,
                 additional_info=None):
        super(MLMUp, self).__init__(
            data_path=data_path,
            _config=_config,
        )

        self.mlm_prob = _config.mlm_prob
        self.drug_name_replace_prob = _config.drug_name_replace_prob
        self.smi_token_id = _config.smi_token_id

    @staticmethod
    def convert_data(series: pd.Series, self):
        """
        处理读入数据，输出tokenizer后的输入数据与样本标签。
        Args:
          series (:obj:`pd.Series`):
            DataFrame的行
          self
        Returns:
          tokenized (:obj:`List[int]`):
            MLM任务的标签id序列，未对齐
          tokenized_mlm (:obj:`List[int]`):
            MLM任务的输入id序列，未对齐
        """
        mlm_prob = self.mlm_prob
        smi_token_id = self.smi_token_id
        mask_token_id = self.tokenizer_txt.mask_token_id

        drug_name = series['DRUG_NAME'].lower()
        abstract = series['ABSTRACTS'].lower()
        smi_tokenized = self.tokenizer_smi.encode(series['C_SMILES'], add_special_tokens=False)

        # 将药物名称概率的替换为'[SMI]'，以便下一步替换为分子式。
        if random.random() < self.drug_name_replace_prob:
            abstract = abstract.replace(drug_name, '[SMI]')
        abstract_tokenized = self.tokenizer_txt.encode(abstract, add_special_tokens=False)

        # 将'[SMI]'的位置替换为对应的分子式的ID序列，并且概率的将token替换为'[MASK]'对应的ID。
        tokenized = []
        tokenized_mlm = []
        for token_id in abstract_tokenized:
            if token_id == smi_token_id:
                for smi_id in smi_tokenized:
                    if random.random() < mlm_prob:
                        tokenized_mlm.append(mask_token_id)
                    else:
                        tokenized_mlm.append(smi_id)
                    tokenized.append(smi_id)
            if random.random() < mlm_prob:
                tokenized_mlm.append(mask_token_id)
            else:
                tokenized_mlm.append(token_id)
            tokenized.append(token_id)

        return tokenized, tokenized_mlm

    def convert_dataset(self):
        """
        将原始输入语料转换为模型的输入
        Args:
          self
        Returns:
          :obj:`Dict`: 字典Key的描述
        """
        tmp_data = pd.DataFrame()
        tqdm.pandas(desc='Tokenize&随机掩模中。。。。')
        tmp_data[['tokenized', 'tokenized_mlm']] = self.data.progress_apply(self.convert_data, axis=1, args=(self,),
                                                                            result_type='expand')
        tmp_tokenized_list = tmp_data['tokenized'].tolist()
        tmp_tokenized_mlm_list = tmp_data['tokenized_mlm'].tolist()
        del tmp_data

        # 将不同语句的token的id展开到同一个list中，并转换为tensor且对齐
        tmp_tokenized_list = [t_id for t_ids in tmp_tokenized_list for t_id in t_ids]
        # tmp_tokenized_list += [self.tokenizer_txt.pad_token_id] * (self.max_seq_len - len(tmp_tokenized_list)%self.max_seq_len)
        tmp_tokenized_list = tmp_tokenized_list[:-(len(tmp_tokenized_list) % (self.max_seq_len - 2))]
        tmp_tokenized_tensor = torch.tensor(tmp_tokenized_list).long().reshape((-1, (self.max_seq_len - 2)))
        del tmp_tokenized_list

        # 将不同语句的token的id展开到同一个list中，并转换为tensor且对齐
        tmp_tokenized_mlm_list = [t_id for t_ids in tmp_tokenized_mlm_list for t_id in t_ids]
        # tmp_tokenized_mlm_list += [self.tokenizer_txt.pad_token_id] * (self.max_seq_len - len(tmp_tokenized_mlm_list)%self.max_seq_len)
        tmp_tokenized_mlm_list = tmp_tokenized_mlm_list[:-(len(tmp_tokenized_mlm_list) % (self.max_seq_len - 2))]
        tmp_tokenized_mlm_tensor = torch.tensor(tmp_tokenized_mlm_list).long().reshape((-1, (self.max_seq_len - 2)))
        del tmp_tokenized_mlm_list

        # 为对齐后的输入与标签tensor添加special token
        cls_token_tensor = torch.tensor(self.tokenizer_txt.cls_token_id).repeat(tmp_tokenized_tensor.shape[0], 1)
        sep_token_tensor = torch.tensor(self.tokenizer_txt.sep_token_id).repeat(tmp_tokenized_tensor.shape[0], 1)
        tmp_tokenized_tensor = torch.cat((cls_token_tensor, tmp_tokenized_tensor, sep_token_tensor), dim=1)
        tmp_tokenized_mlm_tensor = torch.cat((cls_token_tensor, tmp_tokenized_mlm_tensor, sep_token_tensor), dim=1)

        return {
            'input_ids': tmp_tokenized_mlm_tensor,
            'token_type_ids': torch.zeros_like(tmp_tokenized_mlm_tensor, dtype=torch.long),
            'attention_mask': torch.ones_like(tmp_tokenized_mlm_tensor, dtype=torch.float),
            'labels': tmp_tokenized_tensor,
        }


class ClintoxUp(BaseUp):
    """

    Args:
      self

      data_path

      _config(:obj:`Config`):
        tokenizer_txt_type赋值为'none'
    """

    def __init__(self,
                 data_path,
                 _config):
        super(ClintoxUp, self).__init__(
            data_path=data_path,
            _config=_config,
        )

    @staticmethod
    def convert_data(series: pd.Series, self):
        inputs = self.tokenizer_smi(series['ids'], padding='max_length', truncation=True, max_length=self.max_seq_len,
                                    return_attention_mask=True, return_token_type_ids=True)
        if 'train' == self.mode:
            labels = int(series['y1'])
        else:
            labels = -1

        return inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], labels

    def convert_dataset(self):
        tmp_data = pd.DataFrame()
        tqdm.pandas(desc='数据转换中。。。。')
        tmp_data[['input_ids', 'token_type_ids', 'attention_mask', 'labels']] = self.data.progress_apply(
            self.convert_data, axis=1, args=(self,), result_type='expand')

        return {
            'input_ids': tmp_data['input_ids'],
            'token_type_ids': tmp_data['token_type_ids'],
            'attention_mask': tmp_data['attention_mask'],
            'labels': tmp_data['labels'],
        }


class BaseDataset(Dataset):
    def __init__(self,
                 dataset,
                 mode='train'):
        self.mode = mode
        self.nums = dataset['input_ids'].shape[0]

        self.input_ids = dataset['input_ids']
        self.token_type_ids = dataset['token_type_ids']
        self.attention_mask = dataset['attention_mask']
        self.labels = dataset['labels']

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        pass


class MLMDataset(BaseDataset):
    def __init__(self,
                 dataset,
                 _config,
                 additional_info=None):
        super(MLMDataset, self).__init__(
            dataset,
            _config.mode,
        )

    def __getitem__(self, item):
        data = {
            'input_ids': self.input_ids[item],
            'token_type_ids': self.token_type_ids[item],
            'attention_mask': self.attention_mask[item],
            'labels': self.labels[item]
        }

        return data


class ClintoxDataset(BaseDataset):
    def __init__(self,
                 dataset,
                 _config,
                 additional_info=None):
        super(ClintoxDataset, self).__init__(
            dataset,
            _config.mode,
        )

    def __getitem__(self, item):
        data = {
            'input_ids': torch.tensor(self.input_ids[item], dtype=torch.long),
            'token_type_ids': torch.tensor(self.token_type_ids[item], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[item], dtype=torch.float),
        }
        if 'train' == self.mode:
            data['labels'] = torch.tensor(self.labels[item], dtype=torch.float).unsqueeze_(dim=0)

        return data


if __name__ == '__main__':
    # 测试MLMUp函数
    config = Config('MLM')
    data_up = MLMUp(os.path.join(base_dir, config.data_dir, config.pre_train_corpus_file), config)
    dataset = data_up.convert_dataset()
    ic(dataset['input_ids'].shape)
    ic(dataset['labels'].shape)

    # 测试ClintoxUp函数
    config = Config('DT_1', 'train')
    data_up = ClintoxUp(os.path.join(base_dir, config.data_dir, config.downstream_tasks_corpus_file['DT_1']['test']), config)
    dataset = data_up.convert_dataset()
    ic(dataset['input_ids'].shape)
    ic(dataset['labels'].shape)
    # ic(dataset['labels'])

    # 测试MLMDataset类
    mlmdataset = MLMDataset(dataset, config)
    for data in mlmdataset:
      ic(data)
      break


