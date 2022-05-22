class Config:
    def __init__(self,
                 task_type,
                 mode,
                 scale,
                 use_pre_converted_data,
                 num_workers,
                 gpu_nums,
                 dist_url='env://',
                 accum_steps=1,
                 drug_name_replace_prob=0.6):
        # 常量
        self.smi_token_id = 28895
        self.len_of_tokenizer = 28895 + 3117 + 1

        # 训练规模
        self.gpu_num = gpu_nums
        self.scale = scale.lower()
        self.use_pre_converted_data = use_pre_converted_data
        self.num_workers = num_workers
        if 'cpu_mini' == self.scale:
            self.gpu_ids = '-1'

            self.pin_memory = False
            self.pre_train_batch_size = 1
            self.pre_train_epochs = 1
            self.train_batch_size = 8
            self.train_epochs = 10
            self.show_results_times = 2

            self.pre_train_corpus_file_path = 'pre_train/pretrain_corpus_n.csv'

        elif 'gpu_mini' == self.scale:
            self.gpu_ids = '0'

            self.pin_memory = False
            self.pre_train_batch_size = 1
            self.pre_train_epochs = 30
            self.train_batch_size = 8
            self.train_epochs = 1
            self.show_results_times = 10

            self.pre_train_corpus_file_path = 'pre_train/pre_train_corpus_small.csv'
        elif 'gpu_mid' == self.scale:
            self.gpu_ids = '0'

            self.pin_memory = True
            self.pre_train_batch_size = 8
            self.pre_train_epochs = 10
            self.train_batch_size = 16
            self.train_epochs = 20
            self.show_results_times = 10

            self.pre_train_corpus_file_path = 'pre_train/pre_train_corpus_big.csv'

        elif 'gpu_mul' == self.scale:
            self.gpu_ids = ','.join([str(i) for i in range(gpu_nums)])
            self.dist_url = dist_url

            self.pin_memory = True
            self.accum_steps = accum_steps
            self.pre_train_batch_size = 512 // accum_steps
            self.pre_train_epochs = 4
            self.train_batch_size = 16
            self.train_epochs = 40
            self.show_results_times = 10

            self.pre_train_corpus_file_path = 'preprocess/tokenized_data_only_single_gpu_mid_0.6.csv'

        # 训练参数
        self.max_seq_len = 128
        self.ignore_index = -100
        self.model_save_steps = 2000

        self.mlm_prob = 0.15
        self.max_prediction_per_seq = round(self.max_seq_len * self.mlm_prob)
        self.mlm_replace_mask_prob = 0.8
        self.mlm_replace_random_prob = 0.1
        self.drug_name_replace_prob = drug_name_replace_prob

        self.mid_linear_dims = 128
        self.dropout_prob = 0.1

        self.lr = 2e-5 * gpu_nums if gpu_nums else 2e-5
        self.other_lr = 2e-5 * gpu_nums if gpu_nums else 2e-5
        self.weight_decay = 0.01
        self.other_weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_proportion = 0.1

        # 目录
        self.bert_dir = 'models/PubMedBERT_abstract'
        self.data_dir = 'data/'
        self.out_model_dir = 'models/'
        self.spe_file = 'SPE_ChEMBL.txt'
        self.spe_voc_file = 'spe_voc.txt'
        self.converted_pre_train_courpus_path = 'pre_train/converted_pre_train_corpus_bert_raw.pkl'

        self.downstream_tasks_corpus_file = {
            'DT1-1': {
                'train': 'DT1/T1_clintox/train_clintox.csv',
                'valid': 'DT1/T1_clintox/valid_clintox.csv',
                'test': 'DT1/T1_clintox/test_clintox.csv'
            },
            'DT1-2': {
                'train': 'DT1/T2_tox21/train_tox21.csv',
                'valid': 'DT1/T2_tox21/valid_tox21.csv',
                'test': 'DT1/T2_tox21/test_tox21.csv'
            },
            'DT1-3': {
                'train': 'DT1/T3_hiv/train_hiv.csv',
                'valid': 'DT1/T3_hiv/valid_hiv.csv',
                'test': 'DT1/T3_hiv/test_hiv.csv'
            },
            'DT1-4': {
                'train': 'DT1/T4_sider/train_sider.csv',
                'valid': 'DT1/T4_sider/valid_sider.csv',
                'test': 'DT1/T4_sider/test_sider.csv'
            },
            'DT1-5': {
                'train': 'DT1/T5_bbbp/train_bbbp.csv',
                'valid': 'DT1/T5_bbbp/valid_bbbp.csv',
                'test': 'DT1/T5_bbbp/test_bbbp.csv'
            },
            'DT1-6': {
                'train': 'DT1/T6_toxcast/train_toxcast.csv',
                'valid': 'DT1/T6_toxcast/valid_toxcast.csv',
                'test': 'DT1/T6_toxcast/test_toxcast.csv'
            },
            'DT1-7': {
                'train': 'DT1/T7_muv/train_muv.csv',
                'valid': 'DT1/T7_muv/valid_muv.csv',
                'test': 'DT1/T7_muv/test_muv.csv'
            },
            'DT1-8': {
                'train': 'DT1/T8_bace/train_bace.csv',
                'valid': 'DT1/T8_bace/valid_bace.csv',
                'test': 'DT1/T8_bace/test_bace.csv'
            },
        }

        # 类型选择
        self.tokenizer_txt_type = 'default'
        self.tokenizer_smi_type = 'default'
        self.bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        self.task_type = task_type.upper()
        self.mode = mode.lower()
        self.pre_train_task = 'MLM'

    def show_train_parameters(self):
        if 'DT' in self.task_type:
            param_dict = {
                'data': {
                    'task_type': self.task_type
                },
            }
        else:
            param_dict = {
                'data': {
                    'max_seq_len': self.max_seq_len,
                    'mlm_prob': self.mlm_prob,
                    'drug_name_replace_prob': self.drug_name_replace_prob,
                },
            }

        param_dict['model'] = {
            'mid_linear_dims': self.mid_linear_dims,
            'dropout_prob': self.dropout_prob,
        }
        param_dict['train'] = {
            'lr': self.lr,
            'other_lr': self.other_lr,
            'weight_decay': self.weight_decay,
            'other_weight_decay': self.other_weight_decay,
            'adam_epsilon': self.adam_epsilon,
            'warmup_proportion': self.warmup_proportion,
        }

        return param_dict

    def show_train_info(self):
        return {
            'pre_train_batch_size': self.pre_train_batch_size,
            'pre_train_epochs': self.pre_train_epochs,
            'train_batch_size': self.train_batch_size,
            'train_epochs': self.train_epochs,
            'model_save_steps': self.model_save_steps,
            'pre_train_corpus_file_path': self.pre_train_corpus_file_path,
            'bert_name': self.bert_name,
        }
