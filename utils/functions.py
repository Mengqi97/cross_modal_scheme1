import os
import sys
import collections
import codecs
from typing import List, Optional

from config import Config

import torch
from SmilesPE.tokenizer import SPE_Tokenizer
from transformers import PreTrainedTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from loguru import logger

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)


def load_vocab(vocab_file):
    """
    读取静态存储的spe切分字典，该字典的token的id为了与PubMedBERT的字典合并，整体进行了偏移。并且Special Token的ID采用和PubMedBERT相同
    :param vocab_file: 字典的路径
    :return:
      vocab (:obj:'dict', )({token:id, ...}):
    """
    vocab = collections.OrderedDict()  # OrderedDict是为了兼容py3.6之前的python，这时的python字典是无序的。
    with open(vocab_file, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    for line in lines:
        token, index = line.split(' ')
        vocab[token] = int(index.rstrip("\n"))
    return vocab


# 定义SMILES的Tokenizer
class SMILES_SPE_Tokenizer(PreTrainedTokenizer):
    r"""
    Constructs a SMILES tokenizer. Based on SMILES Pair Encoding (https://github.com/XinhaoLi74/SmilesPE).
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        spe_file (:obj:`string`):
            File containing the trained SMILES Pair Encoding vocabulary.
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    def __init__(
            self,
            vocab_file,
            spe_file,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file)
            )
        if not os.path.isfile(spe_file):
            raise ValueError(
                "Can't find a SPE vocabulary file at path '{}'.".format(spe_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.spe_vocab = codecs.open(spe_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.spe_tokenizer = SPE_Tokenizer(self.spe_vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        return self.spe_tokenizer.tokenize(text).split(' ')

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path, filename_prefix: Optional[str] = None):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
            filename_prefix:
                父类的默认参数
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, 'vocab.txt')
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


def load_model_and_parallel(model, gpu_ids: str, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到GPU中（单卡/多卡）
    :param model: 已创建模型。
    :param gpu_ids: 使用的GPU的id值，如：’0,1‘。
    :param ckpt_path: 存储好的模型的地址。
    :param strict: torch.load的参数。
    :return: 模型以及所用设备（cuda）
    """

    gpu_ids = gpu_ids.split(',')

    # set to device to first cuda
    device = torch.device('cpu' if gpu_ids[0] == '-1' else 'cuda:' + gpu_ids[0])

    # 从本地载入模型。
    if ckpt_path is not None:
        # logger.info(f'Load ckpt from {ckpt_path}')
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    # 将模型送入指定设备
    model.to(device)

    # 并行
    if len(gpu_ids) > 1:
        # logger.info(f'Use multi gpus in: {gpu_ids}')
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        # logger.info(f'Use single gpu in: {gpu_ids}')
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device


def build_optimizer_and_scheduler(_config: Config, _model, total_train_items):
    """
    创建优化器和调度器
    :param _config: 参数
    :param _model: 模型
    :param total_train_items: 总的迭代次数
    :return: 优化器和迭代器。
    """
    module = (
        _model.module if hasattr(_model, 'module') else _model
    )

    # 差分学习率
    no_decay = ['bias', 'LayerNorm.weight']
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert，差分学习率
        {
            'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': _config.weight_decay,
            'lr': _config.lr,
        },
        {
            'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': _config.lr,
        },

        # 其他模块，差分学习率
        {
            'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': _config.other_weight_decay,
            'lr': _config.other_lr,
        },
        {
            'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': _config.other_lr,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=_config.lr, eps=_config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(_config.warmup_proportion * total_train_items),
        num_training_steps=total_train_items,
    )

    return optimizer, scheduler


def save_model(_config: Config, model, global_step=-1):
    """
    保存模型
    :param _config: 参数
    :param model: 训练好的模型
    :param global_step: 保存时迭代次数
    :return: 无返回
    """
    base_output_dir = os.path.join(base_dir, _config.out_model_dir, _config.task_type)
    output_dir = os.path.join(base_output_dir, 'checkpoint-{:0>5}'.format(global_step)) if \
        global_step > 0 else os.path.join(base_output_dir, 'best')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 并行训练的模型
    model_to_save = (
        model.module if hasattr(model, 'module') else model
    )
    logger.info(f'保存模型：{output_dir}')
    # ic(f'保存模型：{output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

    return os.path.join(output_dir, 'model.pt')
