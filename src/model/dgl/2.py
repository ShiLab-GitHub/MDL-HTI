import pandas as pd
import torch
import datetime
# def return_one_hot(id):
#     herb_code_df = pd.read_csv('../../../data/ddb/idcodes_cluster.txt', sep="\t", header=None)
#     herb_to_id = dict(zip(herb_code_df.iloc[:, 0], herb_code_df.iloc[:, 1]))
#     m = herb_to_id.get(id)
#     if m is None:
#         return torch.zeros(19934)
#     else:
#         m = m.strip("[]")  # 去除字符串中的 "[" 和 "]"
#         return torch.tensor([float(x) for x in m.split(",")])
# print(return_one_hot(1).shape)
now = datetime.datetime.now()
save_folder = "results/" + '_'.join([str(now.day), str(now.month)]) + str(now.strftime("%H:%M:%S"))+'/'

print(save_folder)