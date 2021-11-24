import os
import argparse
import sys

from loguru import logger

from config import Config
from utils.data_handler import MLMUp

base_dir = os.path.dirname(__file__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_type', dest='task_type', default='MLM', type=str)
    parser.add_argument('-m', '--mode', dest='mode', default='train', type=str)
    parser.add_argument('-s', '--scale', dest='scale', default='cpu_mini', type=str)
    parser.add_argument('-p', '--use_pre_converted_data', dest='use_pre_converted_data', default='0', type=int)
    parser.add_argument('--num_workers', dest='num_workers', default='1', type=int)
    parser.add_argument('--gpu_nums', dest='gpu_nums', default='1', type=int)
    parser.add_argument('--drug_name_replace_prob', dest='drug_name_replace_prob', default=0.6, type=float)

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
        gpu_nums=args.gpu_nums,
        drug_name_replace_prob=args.drug_name_replace_prob,
    )

    MLMUp(data_path=os.path.join(base_dir, config.data_dir, config.pre_train_corpus_file_path),
          _config=config).tokenize_data()
