import os
import pickle
import time
import argparse
import sys

from torch.cuda.memory import empty_cache

from config import Config
from utils.data_handler import MLMUp, MLMDataset
from utils.functions import load_model_and_parallel_ddp, build_optimizer_and_scheduler, save_model_ddp, setup, cleanup

import torch
from loguru import logger
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import get_device_name
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import BertForMaskedLM

base_dir = os.path.dirname(__file__)
str_time = time.strftime('[%Y-%m-%d]%H-%M')


# logger.add(os.path.join(base_dir, 'log', f'{str_time}.log'), encoding='utf-8')


def train(rank, word_size, _config: Config):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    logger.info(f"---- Running basic DDP example on rank {rank}. ----")
    setup(rank, word_size)

    if rank == 0:
        logger.info('**********1-1 构建预训练数据集**********')
    if _config.use_pre_converted_data:
        if rank == 0:
            logger.info('使用预处理好的数据')
        with open(os.path.join(
                base_dir,
                _config.data_dir,
                _config.converted_pre_train_courpus_path,
        ), 'rb') as f:
            train_dataset_up = pickle.load(f)
    else:
        if rank == 0:
            logger.info('重新预处理数据')
        train_dataset_up = MLMUp(data_path=os.path.join(base_dir, _config.data_dir, _config.pre_train_corpus_file_path),
                                 _config=_config).convert_dataset()
        if 'convert_data' == _config.mode:
            if rank == 0:
                logger.info('********** 预处理数据结束 **********')
            return

    train_dataset = MLMDataset(
        train_dataset_up,
        _config=_config,
    )

    if rank == 0:
        logger.info('**********1-2 预训练数据集加载器初始化**********')
    train_sampler = DistributedSampler(train_dataset)
    train_batch_sampler = BatchSampler(train_sampler, _config.pre_train_batch_size, drop_last=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_sampler=train_batch_sampler,
                              pin_memory=_config.pin_memory,
                              num_workers=_config.num_workers)
    del train_dataset

    if rank == 0:
        logger.info('**********2-1 模型初始化**********')
    if _config.bert_dir:
        model = BertForMaskedLM.from_pretrained(os.path.join(base_dir, _config.bert_dir))
    else:
        model = BertForMaskedLM.from_pretrained(_config.bert_name)
    model.resize_token_embeddings(_config.len_of_tokenizer)

    model = load_model_and_parallel_ddp(model, rank)

    if rank == 0:
        logger.info(get_device_name(torch.device('cuda:0')))

    if rank == 0:
        logger.info('**********3-1 损失函数**********')
        logger.info('使用BertForMaskedLM自带的损失函数')

    if rank == 0:
        logger.info('**********3-2 优化器**********')
    # 计算训练的总次数
    per_epoch_items = len(train_loader)
    total_train_items = _config.pre_train_epochs * per_epoch_items
    optimizer, scheduler = build_optimizer_and_scheduler(_config, model, total_train_items)

    if rank == 0:
        logger.info('**********4-1 初始化训练参数**********')
    global_step = 0
    items_show_results = per_epoch_items // _config.one_epoch_show_results_times
    # epoch_loss = []
    if rank == 0:
        logger.info('**********4-2 显示训练参数**********')
        logger.info(f'********** 训练规模：{_config.scale} **********')
        for para_type, para_dict in _config.show_train_parameters().items():
            logger.info(f'********** Parameters: {para_type} **********')
            for para_name, para_value in para_dict.items():
                logger.info('{:>30}:  {:>10}'.format(para_name, para_value))

        for info_name, info_value in _config.show_train_info().items():
            logger.info('{:>30}:  {:>10}'.format(info_name, info_value))

        logger.info(f'********** Parameters: scale **********')
        logger.info('{:>30}:  {:>10}'.format('per_epoch_items', per_epoch_items))
        logger.info('{:>30}:  {:>10}'.format('total_train_items', total_train_items))
        logger.info('{:>30}:  {:>10}'.format('items_show_results', items_show_results))

    logger.info('**********5-1 模型训练**********')
    for epoch in range(_config.pre_train_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        mean_loss = torch.zeros(1).to(rank)
        for step, batch_data in enumerate(train_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(rank)
            outputs = model(**batch_data)

            model.zero_grad()

            loss = outputs.loss
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            if rank == 0:
                if not step % items_show_results:
                    logger.info(
                        'Step: {:>10} ---------- MeanLoss: {:>20.15f}'.format(step, mean_loss.item()))

        if rank == 0:
            logger.info('Epoch: {:>5} ---------- Loss: {:>20.15f}'.format(epoch, mean_loss.item()))

            logger.info('**********6-1 模型保存**********')
            save_model_ddp(_config, model)
        dist.barrier()
        
    empty_cache()
    if rank == 0:
        logger.info('**********7 训练结束**********')
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_type', dest='task_type', default='MLM', type=str)
    parser.add_argument('-m', '--mode', dest='mode', default='train', type=str)
    parser.add_argument('-s', '--scale', dest='scale', default='cpu_mini', type=str)
    parser.add_argument('-p', '--use_pre_converted_data', dest='use_pre_converted_data', default=0, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=1, type=int)
    parser.add_argument('--word_size', dest='word_size', default=1, type=int)
    parser.add_argument('--dist_url', dest='dist_url', default='env://', type=str)

    task_type_list = ['MLM']
    mode_list = ['train', 'convert_data']
    scale_list = ['cpu_mini', 'gpu_mini', 'gpu_mid', 'gpu_mul']

    args = parser.parse_args()
    if args.task_type not in task_type_list or \
            args.mode not in mode_list or \
            args.scale not in scale_list:
        logger.info('********** 参数错误 **********')
        sys.exit()
    # logger.info(args.mode)
    # logger.info(args.scale)   

    config = Config(
        task_type=args.task_type,
        mode=args.mode,
        scale=args.scale,
        use_pre_converted_data=False if 0 == args.use_pre_converted_data else True,
        num_workers=args.num_workers,
        gpu_nums=args.word_size,
        dist_url=args.dist_url,
    )

    logger.info(os.system("nvidia-smi"))
    # logger.info(torch.cuda.device_count())
    # logger.info(torch.cuda.current_device())

    mp.spawn(train,
             args=(args.word_size, config),
             nprocs=args.word_size,
             join=True)
    logger.info('**********结束训练**********')
