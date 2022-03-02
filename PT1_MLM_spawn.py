from logging import log
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
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import get_device_name
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import BertForMaskedLM
from sklearn.metrics import accuracy_score

base_dir = os.path.dirname(__file__)
str_time = time.strftime('[%Y-%m-%d]%H-%M')


# logger.add(os.path.join(base_dir, 'log', f'{str_time}.log'), encoding='utf-8')


def train(rank, word_size, _config: Config):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    logger.info(f"---- Running basic DDP example on rank {rank}. ----")
    setup(rank, word_size)

    if rank == 0:
        tb_writer = SummaryWriter(os.path.join(base_dir, 'runs', time.strftime("%Y-%m-%d=%H-%M", time.localtime())))
        logger.info('Tensorboard Path: {}'.format(time.strftime("%Y-%m-%d=%H-%M", time.localtime())))   
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
        train_dataset_uper = MLMUp(data_path=os.path.join(base_dir, _config.data_dir, _config.pre_train_corpus_file_path),
                                 _config=_config)
        train_dataset_up = train_dataset_uper.convert_dataset()
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
    optimizer, scheduler = build_optimizer_and_scheduler(_config, model, total_train_items//_config.accum_steps)

    if rank == 0:
        logger.info('**********4-1 初始化训练参数**********')
    global_step = 0
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
        logger.info('{:>30}:  {:>10}'.format('items_show_results', _config.show_results_times))

        logger.info('**********5-1 模型训练**********')
        time_start = time.time()
    for epoch in range(_config.pre_train_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        mean_loss = torch.zeros(1).to(rank)
        if rank == 0:
            predict_list = []
            predict_smi_list = []
            predict_txt_list = []
            label_list = []
            label_smi_list = []
            label_txt_list = []
        for step, batch_data in enumerate(train_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(rank)
            outputs = model(**batch_data)

            loss = outputs.loss / _config.accum_steps
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            if rank == 0:
                labels = batch_data['labels']
                _, predicts = outputs.logits.max(axis=-1)
                masked_token_mask = labels != _config.ignore_index
                smi_token_mask = labels > _config.smi_token_id
                predict_list += predicts[masked_token_mask].cpu().tolist()
                predict_smi_list += predicts[masked_token_mask & smi_token_mask].cpu().tolist()
                predict_txt_list += predicts[masked_token_mask & (~smi_token_mask)].cpu().tolist()
                label_list += labels[masked_token_mask].cpu().tolist()
                label_smi_list += labels[masked_token_mask & smi_token_mask].cpu().tolist()
                logger.info(label_smi_list)
                label_txt_list += labels[masked_token_mask & (~smi_token_mask)].cpu().tolist()
                logger.info(label_txt_list)

            loss.backward()

            global_step += 1
            if (global_step + 1) % _config.accum_steps == 0 or (global_step + 1) == total_train_items:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                if rank == 0:
                    if global_step+1 == _config.accum_steps:
                        logger.info(os.system("nvidia-smi"))
                    if not (global_step + 1) % _config.show_results_times:
                        acc = accuracy_score(predict_list, label_list)
                        if label_smi_list:
                            acc_smi = accuracy_score(predict_smi_list, label_smi_list)
                        else:
                            acc_smi = 0
                        if label_txt_list:
                            acc_txt = accuracy_score(predict_txt_list, label_txt_list)
                        else:
                            acc_txt = 0
                        logger.info(
                            'Step: {:>10} ---------- MeanLoss: {:>20.15f}'.format(step+1, mean_loss.item()))
                        logger.info(
                            'Step: {:>10} ---------- Acc     : {:>20.15f}'.format(step+1, acc * 100))
                        logger.info(
                            'Step: {:>10} ---------- SMI Acc : {:>20.15f}'.format(step+1, acc_smi * 100))
                        logger.info(
                            'Step: {:>10} ---------- TXT Acc : {:>20.15f}'.format(step+1, acc_txt * 100))
                        tb_writer.add_scalar('mean_loss', mean_loss.item(), global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)
                        tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
                        tb_writer.add_scalar('accuracy', acc * 100, global_step)
                        tb_writer.add_scalar('accuracy_smi', acc_smi * 100, global_step)
                        tb_writer.add_scalar('accuracy_txt', acc_txt * 100, global_step)
                        tb_writer.add_scalar('ratio_smi_txt', round(len(predict_smi_list)/len(predict_txt_list), 4) if len(predict_txt_list)>0 else 0, global_step)

                # if (global_step + 1) % _config.model_save_steps == 0 or global_step + 1 == total_train_items:
                #     if rank == 0:
                #         time_end = time.time()
                #         logger.info('训练步数： {:>10} ---------- 训练时长：{:>20.15f}'.format((global_step+1)*_config.gpu_num, time_end-time_start))
                #         logger.info('**********6-1 模型保存**********')
                #         save_model_ddp(_config, model, global_step=(global_step+1)*_config.gpu_num)
                #     dist.barrier()

        if (epoch+1) % 1 == 0:
            if rank == 0:
                time_end = time.time()
                logger.info('训练步数： {:>10} ---------- 训练时长：{:>20.15f}'.format((global_step+1)*_config.gpu_num, time_end-time_start))
                logger.info('**********6-1 模型保存**********')
                save_model_ddp(_config, model, global_step=epoch+1)
            dist.barrier()
        
        if (epoch+1) % 1 == 0:
            if rank == 0:
                logger.info('Epoch: {:>5} ---------- MeanLoss: {:>20.15f}'.format(epoch+1, mean_loss.item()))                
                logger.info('**********5-2 动态掩模**********')          

            train_dataset_up = train_dataset_uper.convert_dataset()
            train_dataset = MLMDataset(train_dataset_up, _config=_config)
            train_sampler = DistributedSampler(train_dataset)
            train_batch_sampler = BatchSampler(train_sampler, _config.pre_train_batch_size, drop_last=True)
            train_loader = DataLoader(dataset=train_dataset,
                                    batch_sampler=train_batch_sampler,
                                    pin_memory=_config.pin_memory,
                                    num_workers=_config.num_workers)
            del train_dataset

    empty_cache()
    if rank == 0:
        logger.info('**********7 训练结束**********')
        tb_writer.close()
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_type', dest='task_type', default='MLM', type=str)
    parser.add_argument('-m', '--mode', dest='mode', default='train', type=str)
    parser.add_argument('-s', '--scale', dest='scale', default='cpu_mini', type=str)
    parser.add_argument('-p', '--use_pre_converted_data', dest='use_pre_converted_data', default=0, type=int)
    parser.add_argument('-a', '--accum_steps', dest='accum_steps', default=0, type=int)
    parser.add_argument('-i', '--information', dest='information', default='', type=str)
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
    logger.info('Information: {}'.format(args.information if args.information else 'None'))

    config = Config(
        task_type=args.task_type,
        mode=args.mode,
        scale=args.scale,
        use_pre_converted_data=False if 0 == args.use_pre_converted_data else True,
        num_workers=args.num_workers,
        gpu_nums=args.word_size,
        dist_url=args.dist_url,
        accum_steps=args.accum_steps,
    )

    logger.info(os.system("nvidia-smi"))
    # logger.info(torch.cuda.device_count())
    # logger.info(torch.cuda.current_device())

    mp.spawn(train,
             args=(args.word_size, config),
             nprocs=args.word_size,
             join=True)
    logger.info('**********结束训练**********')
