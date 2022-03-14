import argparse


class BaseArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # 常量
        self.parser.add_argument('--smi_token_id', type=int, default='28895',
                                 help='[SMI] token id')
        self.parser.add_argument('--len_of_tokenizer', type=int, default=f'{28895 + 3117 + 1}',
                                 help='vocab size')
        self.parser.add_argument('--ignore_index', type=int, default='-100',
                                 help='')

        # 训练规模
        self.parser.add_argument('--gpu_num', type=int, default='0',
                                 help='use gpu number')
        self.parser.add_argument('--scale', type=str, default='cpu_mini',
                                 help='training scale')
        self.parser.add_argument('--num_workers', type=int, default='1',
                                 help='dataloader param')
        self.parser.add_argument('--gpu_ids', type=int, default='-1',
                                 help='id of use gpu')
        self.parser.add_argument('--pin_memory', type=int, default='0',
                                 help='dataloader param')
        self.parser.add_argument('--show_results_times', type=int, default='2',
                                 help='')
        self.parser.add_argument('--accum_steps', type=int, default='1',
                                 help='')

        # 多卡
        self.parser.add_argument('--dist_url', type=str, default='env://',
                                 help='')

        # 训练参数
        self.parser.add_argument('--max_seq_len', type=int, default='128',
                                 help='')
        self.parser.add_argument('--model_save_steps', type=int, default='2000',
                                 help='')
        self.parser.add_argument('--dropout_prob', type=float, default='0.1',
                                 help='')
        self.parser.add_argument('--lr', type=float, default='2e-5',
                                 help='')
        self.parser.add_argument('--other_lr', type=float, default='2e-4',
                                 help='')
        self.parser.add_argument('--weight_decay', type=float, default='0.01',
                                 help='')
        self.parser.add_argument('--other_weight_decay', type=float, default='0.01',
                                 help='')
        self.parser.add_argument('--adam_epsilon', type=float, default='1e-8',
                                 help='')
        self.parser.add_argument('--warmup_proportion', type=float, default='0.1',
                                 help='')

        # 目录
        self.parser.add_argument('--bert_dir', type=str, default='models/PubMedBERT_abstract',
                                 help='')
        self.parser.add_argument('--data_dir', type=str, default='data/',
                                 help='')
        self.parser.add_argument('--out_model_dir', type=str, default='models/',
                                 help='')
        self.parser.add_argument('--spe_file', type=str, default='SPE_ChEMBL.txt',
                                 help='')
        self.parser.add_argument('--spe_voc_file', type=str, default='spe_voc.txt',
                                 help='')

        # 类型选择
        self.parser.add_argument('--tokenizer_txt_type', type=str, default='default',
                                 help='')
        self.parser.add_argument('--tokenizer_smi_type', type=str, default='default',
                                 help='')
        self.parser.add_argument('--bert_name', type=str,
                                 default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                                 help='')
        self.parser.add_argument('--task_type', type=str, default='PT1',
                                 help='')
        self.parser.add_argument('--mode', type=str, default='train',
                                 help='')
        self.parser.add_argument('--pre_train_task', type=str, default='MLM',
                                 help='')


class PTArgs(BaseArgs):
    def __init__(self):
        super(PTArgs, self).__init__()

        # 训练规模
        self.parser.add_argument('--use_pre_converted_data', type=int, default='0',
                                 help='if use pre_converted_data')
        self.parser.add_argument('--pre_train_batch_size', type=int, default='1',
                                 help='')
        self.parser.add_argument('--pre_train_epochs', type=int, default='1',
                                 help='')

        # 训练参数
        self.parser.add_argument('--mlm_prob', type=float, default='0.15',
                                 help='')
        self.parser.add_argument('--mlm_replace_mask_prob', type=float, default='0.8',
                                 help='')
        self.parser.add_argument('--mlm_replace_random_prob', type=float, default='0.1',
                                 help='')
        self.parser.add_argument('--drug_name_replace_prob', type=float, default='0.6',
                                 help='')

        # 文件路径
        self.parser.add_argument('--pre_train_corpus_file_path', type=str, default='pre_train/pre_train_corpus_small.csv',
                                 help='')
        self.parser.add_argument('--converted_pre_train_courpus_path', type=str, default='pre_train/converted_pre_train_corpus_bert_raw.pkl',
                                 help='')

        self.args = self.parser.parse_args()
        self.args.use_pre_converted_data  = bool(self.args.use_pre_converted_data)
        self.args.pin_memory = bool(self.args.pin_memory)
        self.args.max_prediction_per_seq = round(self.args.max_seq_len * self.args.mlm_prob)
        self.args.lr = 2e-5 * self.args.gpu_num if self.args.gpu_num else 2e-5
        self.args.other_lr = 2e-5 * self.args.gpu_num if self.args.gpu_num else 2e-5

        if self.args.scale == 'gpu_mini':
            self.args.gpu_ids = '0'
            self.args.pre_train_epochs = 2
            self.args.show_results_times = 10
        elif self.args.scale == 'gpu_mid':
            self.args.gpu_ids = '0'
            self.args.pin_memory = True
            self.args.pre_train_batch_size = 8
            self.args.pre_train_epochs = 10
            self.args.pre_train_corpus_file_path = 'pre_train/pre_train_corpus_big.csv'
        elif 'gpu_mul' == self.args.scale:
            self.args.gpu_ids = ','.join([str(i) for i in range(self.args.gpu_num)])
            self.args.pre_train_batch_size = self.args.pre_train_batch_size // self.args.accum_steps
            self.args.pin_memory = True
            self.args.pre_train_epochs = 4
            self.args.pre_train_corpus_file_path = 'preprocess/tokenized_data_only_single_gpu_mid_0.6.csv'


class DTArgs(BaseArgs):
    def __init__(self):
        super(DTArgs, self).__init__()

        self.parser.add_argument('--train_batch_size', type=int, default='8',
                                 help='')
        self.parser.add_argument('--train_epochs', type=int, default='10',
                                 help='')

        # 目录
        downstream_tasks_corpus_file = {
            'DT1': {
                'train': 'T1_clintox/clintox_train_data.csv',
                'valid': 'T1_clintox/clintox_valid_data.csv',
                'test': 'T1_clintox/clintox_test_data.csv'
            },
        }

        self.args = self.parser.parse_args()
        self.args.downstream_tasks_corpus_file = downstream_tasks_corpus_file
        self.args.pin_memory = bool(self.args.pin_memory)
        self.args.lr = 2e-5 * self.args.gpu_num if self.args.gpu_num else 2e-5
        self.args.other_lr = 2e-5 * self.args.gpu_num if self.args.gpu_num else 2e-5

        if self.args.scale == 'gpu_mini':
            self.args.gpu_ids = '0'
            self.args.train_batch_size = 8
            self.args.train_epochs = 40
            self.args.show_results_times = 10
        elif self.args.scale == 'gpu_mid':
            self.args.gpu_ids = '0'
            self.args.pin_memory = True
            self.args.train_batch_size = 16
            self.args.train_epochs = 20
            self.args.show_results_times = 10
        elif self.args.scale == 'gpu_mul':
            self.args.gpu_ids = ','.join([str(i) for i in range(self.args.gpu_num)])
            self.args.pin_memory = True
            self.args.train_batch_size = 16
            self.args.train_epochs = 40
            self.args.show_results_times = 10
            self.args.pre_train_corpus_file_path = 'preprocess/tokenized_data_only_single_gpu_mid_0.6.csv'

if __name__ == '__main__':
    # print(PTArgs().args)
    for key, value in vars(PTArgs().args).items():
        print(f'{key}:{value}')



