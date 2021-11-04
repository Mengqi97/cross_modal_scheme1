class Config:
    def __init__(self,
                 task_type,
                 mode,
                 scale, 
                 use_pre_converted_data):
        # 常量
        self.smi_token_id = 28895
        self.len_of_tokenizer = 28895 + 3117 + 1

        # 训练规模
        self.scale = scale.lower()
        self.use_pre_converted_data = use_pre_converted_data
        if 'cpu_mini' == self.scale:
            self.gpu_ids = '-1'
            
            self.num_workers = 1
            
            self.pre_train_batch_size = 1
            self.pre_train_epochs = 1
            self.train_batch_size = 8
            self.train_epochs = 10

            self.pre_train_corpus_file_path = 'pre_train/pre_train_corpus_small.csv'

        elif 'gpu_mini' == self.scale:
            self.gpu_ids = '0'
            
            self.num_workers = 4
            
            self.pre_train_batch_size = 8
            self.pre_train_epochs = 30
            self.train_batch_size = 16
            self.train_epochs = 40


            self.pre_train_corpus_file_path = 'pre_train/pre_train_corpus_small.csv'
        elif 'gpu_mid' == self.scale:
            self.gpu_ids = '0'

            self.num_workers = 4

            self.pre_train_batch_size = 16
            self.pre_train_epochs = 50
            self.train_batch_size = 16
            self.train_epochs = 40
            self.pre_train_corpus_file_path = 'pre_train/pre_train_corpus_big.csv'


        # 训练参数
        self.max_seq_len = 512

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
        self.converted_pre_train_courpus_path = 'pre_train/converted_pre_train_corpus.pkl'
        
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


    def show_train_parameters(self):
        return {
            'data':{
                'max_seq_len': self.max_seq_len,
                'mlm_prob' : self.mlm_prob,
                'drug_name_replace_prob' : self.drug_name_replace_prob,
            },

            'model':{
                'mid_linear_dims' : self.mid_linear_dims,
                'dropout_prob' : self.dropout_prob,
            },

            'train':{
                'lr' : self.lr,
                'other_lr' : self.other_lr,
                'weight_decay' : self.weight_decay,
                'other_weight_decay' : self.other_weight_decay,
                'adam_epsilon' : self.adam_epsilon,
                'warmup_proportion' : self.warmup_proportion,
            }
        }
    
    def show_train_info(self):
        return {
            'pre_train_batch_size' : self.pre_train_batch_size,
            'pre_train_epochs' : self.pre_train_epochs,
            'pre_train_corpus_file_path' : self.pre_train_corpus_file_path,
            'bert_name' : self.bert_name,
        }
