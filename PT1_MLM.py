import os
import time

from config import Config
from utils.data_handler import MLMUp, MLMDataset
from utils.functions import load_model_and_parallel, build_optimizer_and_scheduler, save_model

from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda import get_device_name
from transformers import BertForMaskedLM

base_dir = os.path.dirname(__file__)
str_time = time.strftime('[%Y-%m-%d]%H-%M')
# logger.add(os.path.join(base_dir, 'log', f'{str_time}.log'), encoding='utf-8')



def train(_config):
    logger.info('**********1-1 构建预训练数据集**********')
    train_dataset = MLMDataset(
        MLMUp(data_path=os.path.join(base_dir, _config.data_dir, _config.pre_train_corpus_file),
              _config=_config).convert_dataset(),
        _config=_config,
    )

    logger.info('**********1-2 预训练数据集加载器初始化**********')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=_config.pre_train_batch_size,
                              sampler=RandomSampler(train_dataset),
                              num_workers=_config.num_workers)
    del train_dataset

    logger.info('**********2-1 模型初始化**********')
    if _config.bert_dir:
        model = BertForMaskedLM.from_pretrained(os.path.join(base_dir, _config.bert_dir))
    else:
        model = BertForMaskedLM.from_pretrained(_config.bert_name)
    model.resize_token_embeddings(_config.len_of_tokenizer)
    model, device = load_model_and_parallel(model, _config.gpu_ids)

    try:
        logger.info(get_device_name(device))
    except:
        logger.info('use cpu only')

    logger.info('**********3-1 损失函数**********')
    logger.info('使用BertForMaskedLM自带的损失函数')

    logger.info('**********3-2 优化器**********')
    # 计算训练的总次数
    total_train_items = _config.pre_train_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(_config, model, total_train_items)

    logger.info('**********4-1 初始化训练参数**********')
    global_step = 0
    # epoch_loss = []
    logger.info('**********4-2 显示训练参数**********')
    logger.info(f'********** 训练规模：{ _config.scale } **********')
    for para_type, para_dict in _config.show_train_parameters().items():
        logger.info(f'********** Parameters: {para_type} **********')
        for para_name, para_value in para_dict.items():
            logger.info('{:>30}:  {:>10}'.format(para_name, para_value))

    for info_name, info_value in _config.show_train_info().items():
        logger.info('{:>30}:  {:>10}'.format(info_name, info_value))

    logger.info('**********5-1 模型训练**********')
    for epoch in range(_config.pre_train_epochs):
        model.train()
        for step, batch_data in enumerate(train_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            outputs = model(**batch_data)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # 测试用
            if 'cpu-mini' == _config.scale:
                if global_step % 5:
                    break

        scheduler.step()
        # epoch_loss.append([loss.cpu().detach().numpy().tolist()])
        logger.info('Epoch: {:>5} ---------- Loss: {:>20.15f}'.format(epoch, loss.cpu().detach().numpy().tolist()))

    logger.info('**********6-1 模型保存**********')
    save_model(_config, model)


if __name__ == '__main__':
    config = Config('MLM', 'train', 'gpu-mid')
    train(config)
    logger.info('**********结束训练**********')

