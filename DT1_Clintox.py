import os
import time
import argparse
import sys

from config import Config
from utils.models import ClintoxModel
from utils.validators import ClintoxValidator
from utils.data_handler import ClintoxUp, ClintoxDataset
from utils.functions import load_model_and_parallel, build_optimizer_and_scheduler, save_model

from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda import get_device_name

base_dir = os.path.dirname(__file__)
str_time = time.strftime('[%Y-%m-%d]%H-%M')

# logger.add(os.path.join(base_dir, 'log', f'{str_time}.log'), encoding='utf-8')


def train(_config: Config):
    logger.info('**********1-1 构建数据集**********')
    train_dataset = ClintoxDataset(
        ClintoxUp(
            os.path.join(base_dir, config.data_dir, config.downstream_tasks_corpus_file[config.task_type]['train']),
            config,
        ).convert_dataset(),
        config,
    )
    valid_dataset = ClintoxDataset(
        ClintoxUp(
            os.path.join(base_dir, config.data_dir, config.downstream_tasks_corpus_file[config.task_type]['valid']),
            config,
        ).convert_dataset(),
        config,
    )

    logger.info('**********1-2 数据集加载器初始化**********')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.train_batch_size,
                              sampler=RandomSampler(train_dataset),
                              num_workers=_config.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=config.train_batch_size,
                              sampler=RandomSampler(valid_dataset),
                              num_workers=_config.num_workers)



    logger.info('**********2-1 模型初始化**********')
    model = ClintoxModel(_config=config)
    model, device = load_model_and_parallel(
        model,
        config.gpu_ids,
        os.path.join(
            base_dir,
            config.out_model_dir,
            config.pre_train_task,
            'best/model.pt'
        ),
        False
    )

    validator = ClintoxValidator(valid_loader, device)
    try:
        logger.info(get_device_name(device))
    except:
        pass

    logger.info('**********3-1 损失函数**********')

    logger.info('**********3-2 优化器**********')
    # 计算训练的总次数
    total_train_items = config.pre_train_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(config, model, total_train_items)

    logger.info('**********4-1 初始化训练参数**********')
    global_step = 0

    logger.info('**********5-1 模型训练**********')
    for epoch in range(config.train_epochs):
        model.train()
        for step, batch_data in enumerate(train_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            out, loss = model(**batch_data)
            # logger.info(out)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # 测试用
            if 'cpu_mini' == config.scale:
                if global_step % 5 == 0:
                    break

        scheduler.step()
        logger.info('Epoch: {:>5} ---------- Loss: {:>20.15f}'.format(epoch, loss.cpu().detach().numpy().tolist()))
        logger.info('Epoch: {:>5} ---------- Accuracy: {:>20.15f}'.format(epoch,validator(model)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_type', dest='task_type', default='DT1', type=str)
    parser.add_argument('-m', '--mode', dest='mode', default='train', type=str)
    parser.add_argument('-s', '--scale', dest='scale', default='cpu_mini', type=str)
    parser.add_argument('-p', '--use_pre_converted_data', dest='use_pre_converted_data', default='0', type=int)
    parser.add_argument('--num_workers', dest='num_workers', default='1', type=int)
    parser.add_argument('--gpu_nums', dest='gpu_nums', default='1', type=int)

    args = parser.parse_args()

    config = Config(
        task_type=args.task_type,
        mode=args.mode,
        scale=args.scale,
        use_pre_converted_data=False if 0 == args.use_pre_converted_data else True,
        num_workers=args.num_workers,
        gpu_nums=args.gpu_nums,
    )

    train(config)
    logger.info('**********结束训练**********')


