import os


class Config():
    def __init__(self):
        super(Config, self).__init__()
        self.root_dir = "/remote-home/yrchen"
        #"tasks/fastnlp-relevant/summarization/cnn-dailymail/finished_files/chunked/train_*"
        self.train_data_path = "/remote-home/yrchen/Datasets/CNNDM/finished_files_new/CNNDM.train.json"
        #"tasks/fastnlp-relevant/summarization/cnn-dailymail/finished_files/val.bin"
        self.eval_data_path = "/remote-home/yrchen/Datasets/CNNDM/finished_files_new/CNNDM.val.json"
        #"tasks/fastnlp-relevant/summarization/cnn-dailymail/finished_files/test.bin"
        self.decode_data_path ="/remote-home/yrchen/Datasets/CNNDM/finished_files_new/CNNDM.test.json"
        self.vocab_path = os.path.join(self.root_dir,
                                       "tasks/fastnlp-relevant/summarization/cnn-dailymail/finished_files/vocab")
        self.log_root = os.path.join(self.root_dir, "tasks/fastnlp-relevant/summarization/my-pnt-sum/log/CNNDM")
        self.train_path = None
        self.model_path = None

        # Hyperparameters
        self.hidden_dim = 256
        self.emb_dim = 128
        self.batch_size = 32
        self.max_enc_steps = 400
        self.max_dec_steps = 100
        self.beam_size = 4
        self.min_dec_steps = 35
        #这个要随着不同数据集的变化而变化
        self.vocab_size = 50000

        self.lr = 0.15
        self.adagrad_init_acc = 0.1
        self.rand_unif_init_mag = 0.02
        self.trunc_norm_init_std = 1e-4
        self.max_grad_norm = 2.0 #2.0

        self.pointer_gen = True
        self.is_coverage = True
        self.cov_loss_wt = 1.0

        self.eps = 1e-12
        self.n_epochs = 100

        self.use_gpu = True

        self.lr_coverage = 0.010#0.15
