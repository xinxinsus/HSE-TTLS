import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else'cpu')#
#DEVICE = torch.device('cpu')


# import random
#
# # 针对每组实验设置不同的种子
# num_experiments = 10
#
# seeds = [random.randint(1, 100000) for _ in range(num_experiments)]
#
# for seed in seeds:
#     random.seed(seed)


TORCH_SEED = 129

class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.bert_cache_path = '/media/tiffany/新加卷/PycharmProjects/ubuntu/pretrained_model/bert-base-chinese'
        self.epochs = 20
        self.batch_size = 2
        self.lr = 1e-5
        self.tuning_bert_rate = 1e-5
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.warmup_proportion = 0.1
        self.TORCH_SEED=TORCH_SEED
        self.feat_dim=768

