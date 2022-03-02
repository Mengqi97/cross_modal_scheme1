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
from torch.utils.tensorboard import SummaryWriter

base_dir = os.path.dirname(__file__)
str_time = time.strftime('[%Y-%m-%d]%H-%M')


# logger.add(os.path.join(base_dir, 'log', f'{str_time}.log'), encoding='utf-8')


def train(_config: Config):
    tb_path = os.path.join(base_dir, 'runs-dt', time.strftime("%Y-%m-%d=%H-%M", time.localtime()))
    logger.info(f'TensorBoard save path: {tb_path}')
    tb_writer = SummaryWriter(tb_path)
    logger.info('**********1-1 构建数据集**********')
    train_dataset = ClintoxDataset(
        ClintoxUp(
            os.path.join(base_dir, _config.data_dir, _config.downstream_tasks_corpus_file[_config.task_type]['train']),
            _config,
        ).convert_dataset(),
        _config,
    )
    valid_dataset = ClintoxDataset(
        ClintoxUp(
            os.path.join(base_dir, _config.data_dir, _config.downstream_tasks_corpus_file[_config.task_type]['valid']),
            _config,
        ).convert_dataset(),
        _config,
    )

    logger.info('**********1-2 数据集加载器初始化**********')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=_config.train_batch_size,
                              sampler=RandomSampler(train_dataset),
                              num_workers=_config.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=_config.train_batch_size,
                              shuffle=False,
                              num_workers=_config.num_workers)

    logger.info('**********2-1 模型初始化**********')
    model = ClintoxModel(_config=_config)
    model, device = load_model_and_parallel(
        model,
        _config.gpu_ids,
        os.path.join(
            base_dir,
            _config.out_model_dir,
            _config.pre_train_task,
            'best/model_ddp.pt'
        ),
        False
    )

    use_n_gpus = False
    if hasattr(model, 'module'):
        use_n_gpus = True

    validator = ClintoxValidator(valid_loader, device)
    try:
        logger.info(get_device_name(device))
    except:
        pass

    logger.info('**********3-1 损失函数**********')

    logger.info('**********3-2 优化器**********')
    # 计算训练的总次数
    per_epoch_items = len(train_loader)
    total_train_items = _config.pre_train_epochs * per_epoch_items
    optimizer, scheduler = build_optimizer_and_scheduler(_config, model, total_train_items)

    logger.info('**********4-1 初始化训练参数**********')
    global_step = 0
    items_show_results = per_epoch_items // _config.show_results_times

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
    for epoch in range(_config.train_epochs):
        model.train()
        for step, batch_data in enumerate(train_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            out, loss = model(**batch_data)

            optimizer.zero_grad()
            if use_n_gpus:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            # 测试用
            if 'cpu_mini' == _config.scale:
                if global_step % 5 == 0:
                    break

            if step + 1 % items_show_results == 0:
                tb_writer.add_scalar('loss', loss.item(), global_step)
                logger.info(
                    'Step: {:>5} ---------- Loss: {:>20.15f}'.format(step + 1, loss.item()))

        acc, auc = validator(model)
        tb_writer.add_scalar('accuracy', acc, epoch+1)
        tb_writer.add_scalar('roc_auc', auc, epoch+1)
        logger.info('Epoch: {:>5} ---------- Accuracy: {:>20.15f}'.format(epoch + 1, acc))
        logger.info('Epoch: {:>5} ---------- ROC_AUC:  {:>20.15f}'.format(epoch + 1, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_type', dest='task_type', default='DT1', type=str)
    parser.add_argument('-m', '--mode', dest='mode', default='train', type=str)
    parser.add_argument('-s', '--scale', dest='scale', default='cpu_mini', type=str)
    parser.add_argument('-p', '--use_pre_converted_data', dest='use_pre_converted_data', default='0', type=int)
    parser.add_argument('--num_workers', dest='num_workers', default='1', type=int)
    parser.add_argument('--gpu_nums', dest='gpu_nums', default='0', type=int)

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
