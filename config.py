class Config:
    def __init__(self,
                 task_type,
                 mode='train',
                 scale='cpu-mini'):
        # 常量
        self.smi_token_id = 28895
        self.len_of_tokenizer = 28895 + 3117 + 1

        # 训练规模
        self.scale = scale.lower()
        if 'cpu-mini' == self.scale:
            self.gpu_ids = '-1'
            self.pre_train_batch_size = 1
            self.pre_train_epochs = 1
            self.train_batch_size = 8
            self.train_epochs = 10
        elif 'gpu-mini' == self.scale:
            self.gpu_ids = '0'
            self.pre_train_batch_size = 8
            self.pre_train_epochs = 30
            self.train_batch_size = 16
            self.train_epochs = 40

        # 训练参数
        self.max_seq_len = 320

        self.mlm_prob = 0.3
        self.drug_name_replace_prob = 0.6

        self.mid_linear_dims = 128
        self.dropout_prob = 0.1

        self.lr = 5e-5
        self.other_lr = 5e-5
        self.weight_decay = 0
        self.other_weight_decay = 0
        self.adam_epsilon = 1e-8
        self.warmup_proportion = 0

        # 目录
        self.bert_dir = 'models/PubMedBERT_abstract'
        self.data_dir = 'data/'
        self.out_model_dir = 'models/'
        self.spe_file = 'SPE_ChEMBL.txt'
        self.spe_voc_file = 'spe_voc.txt'
        self.pre_train_corpus_file = 'pre_train/pre_train_corpus_small.csv'
        self.downstream_tasks_corpus_file = {
            'DT_1': {
                'train': 'T1_clintox/clintox_train_data.csv',
                'valid': 'T1_clintox/clintox_valid_data.csv',
                'test': 'T1_clintox/clintox_test_data.csv'
            },
        }

        # 类型选择
        self.tokenizer_txt_type = 'default'
        self.tokenizer_smi_type = 'default'
        self.bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        self.task_type = task_type.upper()
        self.mode = mode.lower()
        self.pre_train_task = 'MLM'
