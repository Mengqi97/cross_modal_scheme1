import os
import sys
import random
import pickle
import json

from config import Config
from utils.functions import SMILES_SPE_Tokenizer, flat_list_rec

import torch
import nltk
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from icecream import ic
from pandarallel import pandarallel
from loguru import logger

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

random.seed(70627)

tokenizer_smi = None
tokenizer_txt = None
drug_name_replace_prob = None
tokenizer_sen = nltk.data.load('tokenizers/punkt/english.pickle')


class BaseUp:
    def __init__(self,
                 data_path,
                 _config: Config):

        self.data_dir = _config.data_dir
        self.scale = _config.scale
        self.len_of_tokenizer = _config.len_of_tokenizer
        self.max_seq_len = _config.max_seq_len
        self.mode = _config.mode
        self.data = pd.read_csv(data_path).dropna()
        self.ignore_index = _config.ignore_index
        self.max_predictions_per_seq = _config.max_prediction_per_seq

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
        elif 'pubmedBERT' == _config.tokenizer_smi_type:
            self.tokenizer_smi = self.tokenizer_txt


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
        self.mlm_replace_mask_prob = _config.mlm_replace_mask_prob
        self.mlm_replace_random_prob = _config.mlm_replace_random_prob
        self.drug_name_replace_prob = _config.drug_name_replace_prob
        self.smi_token_id = _config.smi_token_id
        self.tokenizer_smi_type = _config.tokenizer_smi_type
        if 'convert_data' == _config.mode:
            self.save_converted_dataset_path = os.path.join(
                base_dir,
                _config.data_dir,
                _config.converted_pre_train_courpus_path,
            )
        else:
            self.save_converted_dataset_path = ''

        del _config

    @staticmethod
    def convert_data(series: pd.Series, self, mode, ignore_index):
        """
        处理读入数据，输出tokenizer后的输入数据与样本标签。
        Args:
          series (:obj:`pd.Series`):
            DataFrame的行
          self:
            pass
          mode:
            是否已经tokenize过
          ignore_index:
            label中的非mask的token的id
        Returns:
          tokenized (:obj:`List[int]`):
            MLM任务的标签id序列，未对齐
          tokenized_mlm (:obj:`List[int]`):
            MLM任务的输入id序列，未对齐
        """
        mlm_prob = self.mlm_prob
        mlm_replace_mask_prob = self.mlm_replace_mask_prob
        mlm_replace_random_prob = self.mlm_replace_mask_prob + self.mlm_replace_random_prob
        len_of_tokenizer = self.len_of_tokenizer
        smi_token_id = self.smi_token_id
        mask_token_id = self.tokenizer_txt.mask_token_id

        if mode == 'raw':
            drug_name = series['DRUG_NAME'].lower()
            abstract = series['ABSTRACTS'].lower()
            smi_tokenized = self.tokenizer_smi.encode(series['C_SMILES'], add_special_tokens=False)

            # 将药物名称概率的替换为'[SMI]'，以便下一步替换为分子式。
            if random.random() < self.drug_name_replace_prob:
                abstract = abstract.replace(drug_name, '[SMI]')
            abstract_tokenized = self.tokenizer_txt.encode(abstract, add_special_tokens=False)
        elif mode == 'tokenized':
            smi_tokenized = json.loads(series['tokenized_smi'])
            abstract_tokenized = json.loads(series['tokenized_txt'])

        # 将'[SMI]'的位置替换为对应的分子式的ID序列，并且概率的将token替换为'[MASK]'对应的ID。
        smi_token_num = abstract_tokenized.count(smi_token_id)
        abstract_txt_len = len(abstract_tokenized) - smi_token_num
        abstract_smi_len = len(smi_tokenized)
        abstract_len = abstract_txt_len + abstract_smi_len * smi_token_num
        masked_txt_num = round(abstract_txt_len * mlm_prob)
        masked_smi_num = [round(abstract_smi_len * mlm_prob) for _ in range(smi_token_num)]
        selected_pos = []
        tokenized_label = [ignore_index] * abstract_len
        tokenized_mlm = [smi_tokenized if ele == smi_token_id else ele for ele in abstract_tokenized]
        tokenized_mlm = flat_list_rec(tokenized_mlm)
        assert len(tokenized_mlm) == abstract_len, '掩模过程出错'
        while masked_txt_num or [num for num in masked_smi_num if num > 0]:
            token_pos = 0
            nth_smi = -1
            for token_id in abstract_tokenized:
                if token_id == smi_token_id:
                    nth_smi += 1
                    for smi_id in smi_tokenized:
                        if masked_smi_num[nth_smi] and (token_pos not in selected_pos) and (random.random() <= mlm_prob):
                            masked_smi_num[nth_smi] -= 1
                            selected_pos.append(token_pos)
                            tmp_replace_prob = random.random()
                            if tmp_replace_prob <= mlm_replace_mask_prob:
                                tokenized_mlm[token_pos] = mask_token_id
                            elif tmp_replace_prob <= mlm_replace_random_prob:
                                tokenized_mlm[token_pos] = random.randint(0, len_of_tokenizer - 1)
                            else:
                                tokenized_mlm[token_pos] = smi_id
                            tokenized_label[token_pos] = smi_id
                        token_pos += 1
                    continue
                if masked_txt_num and (token_pos not in selected_pos) and (random.random() <= mlm_prob):
                    masked_txt_num -= 1
                    selected_pos.append(token_pos)
                    tmp_replace_prob = random.random()
                    if tmp_replace_prob <= mlm_replace_mask_prob:
                        tokenized_mlm[token_pos] = mask_token_id
                    elif tmp_replace_prob <= mlm_replace_random_prob:
                        tokenized_mlm[token_pos] = random.randint(0, len_of_tokenizer - 1)
                    else:
                        tokenized_mlm[token_pos] = token_id
                    tokenized_label[token_pos] = token_id
                token_pos += 1
        return tokenized_label, tokenized_mlm

    def convert_dataset(self):
        """
        将原始输入语料转换为模型的输入
        Args:
          self
        Returns:
          :obj:`Dict`: 字典Key的描述
        """
        tmp_data = pd.DataFrame()
        # tqdm.pandas(desc='Tokenize&随机掩模中。。。。')
        columns_list = self.data.columns.to_list()
        apply_mode = 'raw'
        if 'tokenized_smi' in columns_list:
            apply_mode = 'tokenized'
        tmp_data[['tokenized', 'tokenized_mlm']] = self.data.apply(self.convert_data,
                                                                   axis=1,
                                                                   args=(self,
                                                                         apply_mode,
                                                                         self.ignore_index),
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
        ign_token_tensor = torch.tensor(self.ignore_index).repeat(tmp_tokenized_tensor.shape[0], 1)
        tmp_tokenized_tensor = torch.cat((ign_token_tensor, tmp_tokenized_tensor, ign_token_tensor), dim=1)
        tmp_tokenized_mlm_tensor = torch.cat((cls_token_tensor, tmp_tokenized_mlm_tensor, sep_token_tensor), dim=1)

        converted_dataset = {
            'input_ids': tmp_tokenized_mlm_tensor,
            'token_type_ids': torch.zeros_like(tmp_tokenized_mlm_tensor, dtype=torch.long),
            'attention_mask': torch.ones_like(tmp_tokenized_mlm_tensor, dtype=torch.float),
            'labels': tmp_tokenized_tensor,
        }

        if self.save_converted_dataset_path:
            with open(self.save_converted_dataset_path, 'wb') as f:
                pickle.dump(converted_dataset, f, pickle.HIGHEST_PROTOCOL)

            return ''
        else:
            return converted_dataset

    # @staticmethod
    # def item_tokenize(series: pd.Series):
    #     global tokenizer_smi
    #     global tokenizer_txt
    #     global drug_name_replace_prob
    #     drug_name = series['DRUG_NAME'].lower()
    #     abstract = series['ABSTRACTS'].lower()
    #     smi_tokenized = tokenizer_smi.encode(series['C_SMILES'], add_special_tokens=False)
    #
    #     # 将药物名称概率的替换为'[SMI]'，以便下一步替换为分子式。
    #     if random.random() < drug_name_replace_prob:
    #         abstract = abstract.replace(drug_name, '[SMI]')
    #     abstract_tokenized = tokenizer_txt.encode(abstract, add_special_tokens=False)
    #
    #     return json.dumps(smi_tokenized), json.dumps(abstract_tokenized)

    @staticmethod
    def item_tokenize(series: pd.Series):
        global tokenizer_smi
        global tokenizer_txt
        global drug_name_replace_prob
        drug_name = series['DRUG_NAME'].lower()
        abstract = series['ABSTRACTS'].lower()
        if abstract.find(drug_name) == -1:
            return '', ''
        abstract_sentence_list = tokenizer_sen.tokenize(series['ABSTRACTS'])
        smi_tokenized = tokenizer_smi.encode(series['C_SMILES'], add_special_tokens=False)

        abstract_sentence_filter_list = [sentence.lower() for sentence in abstract_sentence_list if
                                         sentence.lower().find(drug_name) != -1]
        abstract_list = []
        # 将药物名称概率的替换为'[SMI]'，以便下一步替换为分子式。
        for sentence in abstract_sentence_filter_list:
            if random.random() < drug_name_replace_prob:
                abstract_list.append(sentence.replace(drug_name, '[SMI]'))
            else:
                abstract_list.append(sentence)
        abstract_tokenized = tokenizer_txt.encode(' '.join(abstract_list), add_special_tokens=False)

        return json.dumps(smi_tokenized), json.dumps(abstract_tokenized)

    def tokenize_data(self):
        tokenized_data = pd.DataFrame()
        logger.info('-- begin --')
        pandarallel.initialize()
        global tokenizer_smi
        global tokenizer_txt
        global drug_name_replace_prob
        tokenizer_smi = self.tokenizer_smi
        tokenizer_txt = self.tokenizer_txt
        drug_name_replace_prob = self.drug_name_replace_prob
        tokenized_data[['tokenized_smi', 'tokenized_txt']] = self.data.parallel_apply(self.item_tokenize,
                                                                                      axis=1,
                                                                                      result_type='expand')
        tokenizer_smi = None
        tokenizer_txt = None
        drug_name_replace_prob = None
        logger.info('-- end --')
        save_path = os.path.join(base_dir, self.data_dir, 'preprocess')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tokenized_data.to_csv(os.path.join(
            save_path,
            f'tokenized_data_only_single_{self.tokenizer_smi_type}_{self.drug_name_replace_prob}.csv',
        ), index=False)


class DT1Up(BaseUp):
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
        super(DT1Up, self).__init__(
            data_path=data_path,
            _config=_config,
        )

    @staticmethod
    def convert_data(series: pd.Series, self):
        inputs = self.tokenizer_smi(series['smiles'], padding='max_length', truncation=True, max_length=self.max_seq_len,
                                    return_attention_mask=True, return_token_type_ids=True)
        if 'train' == self.mode:
            labels = list(series[2:])
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


class DT1Dataset(BaseDataset):
    def __init__(self,
                 dataset,
                 _config,
                 additional_info=None):
        super(DT1Dataset, self).__init__(
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
    data_up = DT1Up(os.path.join(base_dir, config.data_dir, config.downstream_tasks_corpus_file['DT_1']['test']),
                    config)
    dataset = data_up.convert_dataset()
    ic(dataset['input_ids'].shape)
    ic(dataset['labels'].shape)
    # ic(dataset['labels'])

    # 测试MLMDataset类
    mlmdataset = MLMDataset(dataset, config)
    for data in mlmdataset:
        ic(data)
        break
