import numpy as np
import random
import torch
import time
from os import path, mkdir
import shutil
from dateutil.relativedelta import relativedelta
import pickle
import os
import pandas as pd

np.random.seed(0)
random.seed(0)
#torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.equality_patience = 3
        self.verbose = verbose
        self.counter = 0
        self.equality_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model, contexts, save_ctx):
        # if val_loss == 0.0:
        #     self.early_stop = True
        #     return
        if val_loss < 0:
            self.early_stop = True
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, contexts, save_ctx)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # elif score == self.best_score - self.delta:
        #     self.equality_counter += 1
        #     print(f'EarlyStopping counter: {self.equality_counter} out of {self.equality_patience}')
        #     if self.equality_counter >= self.equality_patience:
        #         self.early_stop = True
        # elif val_loss == 0.0:
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model)
        #     self.equality_counter += 1
        #     if self.equality_counter >= self.equality_patience:
        #         self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, contexts, save_ctx)
            self.counter = 0
            self.equality_counter = 0

    def save_checkpoint(self, val_loss, model, contexts, save_ctx):
        """Saves model when validation loss decrease."""
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.save_path)
        import os
        if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
        torch.save(model, self.save_path)
        if save_ctx:
            with open(self.save_path+'.pkl', 'wb') as handle:
                pickle.dump(contexts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.val_loss_min = val_loss

def currentTime(self):
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

def check_n_create(dir_path, overwrite=False):
    if not path.exists(dir_path):
        mkdir(dir_path)
    else:
        if overwrite:
            shutil.rmtree(dir_path)
            mkdir(dir_path)

def create_directory_tree(dir_path):
    for i in range(len(dir_path)):
        check_n_create(path.join(*(dir_path[:i + 1])))

def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)
def return_herb():
    # 设置中药文件夹的路径
    root_dir = "../data/ddb/herb/"
    # 存储所有成分的集合
    all_components_set = set()
    # 存储所有中药的列表
    herbal_medicines_list = []
    # 遍历每个中药文件夹
    for herbal_medicine_folder in os.listdir(root_dir):
        # 将中药名字添加到列表中
        herbal_medicines_list.append(herbal_medicine_folder)
        # 构建相关文件的路径
        related_ingredient_file = os.path.join(root_dir, herbal_medicine_folder, "related_ingredient.xlsx")
        # 读取相关文件中的成分信息
        df = pd.read_excel(related_ingredient_file)
        ingredients = df["related ingredient"].tolist()
        # 将成分信息添加到集合中
        all_components_set.update(ingredients)
    # 将集合转换为有序列表
    all_components_list = list(all_components_set)
    # 确定成分数目
    num_components = len(all_components_list)
    # 创建零向量作为编码结果
    one_hot = np.zeros((len(os.listdir(root_dir)), num_components))
    # 遍历每个中药文件夹
    for i, herbal_medicine_folder in enumerate(os.listdir(root_dir)):
        # 构建相关文件的路径
        related_ingredient_file = os.path.join(root_dir, herbal_medicine_folder, "related_ingredient.xlsx")
        # 读取相关文件中的成分信息
        df = pd.read_excel(related_ingredient_file)
        ingredients = df["related ingredient"].tolist()
        # 对于每个中药，将其包含的成分的位置设置为1
        for ingredient in ingredients:
            component_index = all_components_list.index(ingredient)
            one_hot[i][component_index] = 1
    return one_hot, herbal_medicines_list

