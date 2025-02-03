from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch
import pandas as pd
import dgl
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from .graph_bert_utils import GraphBertConfig, GraphBert_path
from .discriminator import Discriminator
from transformers.models.bert.modeling_bert import BertPreTrainedModel
import json

np.random.seed(0)
random.seed(0)
#torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class DynamicLinearLayer(nn.Module):
    def __init__(self):
        super(DynamicLinearLayer, self).__init__()
        # 定义不同输入尺寸对应的三个线性层及其激活函数
        self.layers_target = nn.Sequential(
            nn.Linear(663, 350),
            nn.ReLU(),
            nn.Linear(350, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.layers_herb = nn.Sequential(
            nn.Linear(19858, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )

    def forward(self, x):
        # 确保输入 x 的形状为 [batch_size, input_size]
        if x.ndim == 1:
            x = x.unsqueeze(0)  # 增加批次维度
        input_size = x.size(1)

        # 根据输入大小选择对应的层组
        if input_size == 663:
            output = self.layers_target(x)
        elif input_size == 19858:
            output = self.layers_herb(x)
        else:
            print(input_size)
            raise ValueError("Unsupported input dimension")  # 如果不支持该维度，抛出异常

        # 移除批次维度，如果它的大小是1
        output = output.squeeze(0)
        return output

def load_index_mapping(file_path):
    try:
        with open(file_path, "r") as file:
            index_2entity = json.load(file)
        return index_2entity
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

index_2entity = load_index_mapping('../data/ddb/herb_data/idx_to_entity.json')

def idx_id(idx, index_2entity=index_2entity):
    if index_2entity is None:
        return "Error loading index mapping."
    idx_str = str(idx)
    if idx_str in index_2entity:
        return float(index_2entity[idx_str])
    else:
        print('index:', idx_str)
        return "Index does not exist."


class WeightedSumModule(nn.Module):
    def __init__(self):
        super(WeightedSumModule, self).__init__()
        # 初始化参数 m 和 n，并标记为可训练参数
        self.m = nn.Parameter(torch.randn(1))
        self.n = nn.Parameter(torch.randn(1))

    def forward(self, A, B):
        # 应用权重并计算加权和
        C = self.m * A + self.n * B
        return C

def return_name(x):
    df = pd.read_csv('../data/ddb/herb_data/entity2.txt', sep="\t", header=None)
    return df.loc[df[df.columns[1]] == x, df.columns[0]].iloc[0]


herb_code_df = pd.read_csv('../data/ddb/herb_data/codes_one_493.txt', sep="\t", header=None)
herb_to_id = dict(zip(herb_code_df.iloc[:, 0], herb_code_df.iloc[:, 1]))

def return_one_hot(id, herb_to_id=herb_to_id):
    m = herb_to_id.get(id)
    if m is not None:
        m = m.strip("[]")  # 去除字符串中的 "[" 和 "]"
        return torch.tensor([float(x) for x in m.split(",")])
    else:
        print("nan:", id)
        return torch.zeros(13278)


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_features):
        super(SelfAttentionLayer, self).__init__()
        # 线性层用于生成查询（queries）、键（keys）和值（values）
        self.query = nn.Linear(input_features, input_features)
        self.key = nn.Linear(input_features, input_features)
        self.value = nn.Linear(input_features, input_features)
        # 线性层用于将注意力机制的输出转换为最终的单个值
        self.output = nn.Linear(input_features, 1)

    def forward(self, x, mask=None):
        # 生成查询（queries）、键（keys）和值（values）
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # 计算注意力分数
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        # 应用掩码，如果提供的话
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 归一化注意力分数
        attention_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重到值（values）上
        attended_values = torch.matmul(attention_weights, values)

        # 将加权的值（attended_values）转换为最终的输出
        output = self.output(attended_values)
        return output



class GraphClassifier(nn.Module):
    def l2_norm(self, x, dim=1):
        return x / (torch.max(torch.norm(x, dim=dim, keepdim=True), self.epsilon))

    def __init__(self, params, dl):
        super().__init__()

        self.params = params
        self.attention_weights = None
        self.gnn = RGCN(params)

        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.hidden_dim, sparse=False)

        # self.fc_layer = nn.Linear(4 * self.params.num_layers * self.params.hidden_dim + 2 * self.params.hidden_dim, 1)
        #self.fc_layer = nn.Linear(2 * self.params.num_layers * self.params.hidden_dim + 2 * self.params.hidden_dim+128+128, 256)
        #self.fc_layer = nn.Linear(768, 1)
        # self.fc_layer3 = nn.Linear(1152, 512, device=self.params.device)
        # self.fc_layer2 = nn.Linear(512, 128, device=self.params.device)
        #self.fc_layer1 = nn.Linear(256, 1, device=self.params.device)
        self.fc_layer = SelfAttentionLayer(2 * self.params.num_layers * self.params.hidden_dim + 2 * self.params.hidden_dim+128+128)

        config = GraphBertConfig(graph_size=dl.args.nodes_all, rel_size=dl.args.rel_size,
                                 edge_size=dl.edges_train['total'],
                                 in_feat=None, node_feat=None, rel_feat=None, edge_feat=None,
                                 node_dims=None, rel_dims=None, node_features=None, rel_features=None,
                                 seq_length=params.max_seq_length, neighbor_samples=params.context_hops,
                                 path_length=params.max_path_len, path_samples=params.path_samples,
                                 _sep_id=dl.args._sep_id, _cls_id=dl.args._cls_id, _pad_id=dl.args._pad_id,
                                 label_rp=dl.args.rel_size, label_lp=2, hidden_size=params.hidden_dim,
                                 edge_hidden_size=params.hidden_dim,
                                 num_hidden_layers=params.num_layers, num_attention_heads=params.num_heads,
                                 hidden_dropout_prob=params.dropout, attention_probs_dropout_prob=params.dropout,
                                 node_id2index=None, node_index2type=None
                                 )
        self.seq_model = GraphBertForPairScoring(config)
        self.proj_ctx = nn.Linear(self.params.hidden_dim * self.params.num_layers, self.params.hidden_dim)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.params.dropout)

        self.Hn = nn.Parameter(torch.FloatTensor(dl.args.nodes_all, self.params.num_clusters))
        self.Zn = nn.Parameter(torch.FloatTensor(self.params.num_clusters, self.params.hidden_dim))
        nn.init.xavier_normal_(self.Hn)
        nn.init.xavier_normal_(self.Zn)
        self.cluster_act = nn.Softmax(dim=-1)
        self.cluster_dropout = nn.Dropout(self.params.cluster_dropout)
        self.summary_act = nn.Sigmoid()
        #self.one_hot_layer = nn.Linear(13801, 4 * 64, device=self.params.device)
        self.one_hot_layer = DynamicLinearLayer()
        #self.out = WeightedSumModule()
        #self.layer_one_hot1 = nn.Linear(2048, 1024, device=self.params.device)
        #self.layer_one_hot2 = nn.Linear(1024, 512, device=self.params.device)
        #self.layer_one_hot3 = nn.Linear(512, 1, device=self.params.device)
        # self.W_c = nn.Bilinear(self.params.hidden_dim, self.params.hidden_dim, self.params.hidden_dim)
        # self.W_ht = nn.Bilinear(self.params.hidden_dim, self.params.hidden_dim, self.params.hidden_dim)

        self.k = torch.tensor(self.params.num_clusters, dtype=torch.float32, device=self.params.device,
                              requires_grad=False)

    def forward(self, data_pos, data_neg, dl, type="t"):
        h1_l_loss, h1_o_loss, h1_s_loss, logit = [], [], [], None
        tokens0, masks, segments = [], [], []
        tokens0_neg, masks_neg, segments_neg = [], [], []
        view_embeds, view_aggr, rep_rel = None, None, None

        adj, I = dl.adj, dl.I
        degrees = dl.edges_train['adj_all'].sum(0).reshape((-1, 1))
        m = degrees.sum()
        degrees = torch.tensor(degrees, dtype=torch.float32, device=self.params.device, requires_grad=False)

        ######################################## positive subgraph modeling ########################################
        ''' enclosing subgraph view learning '''
        g, _, rel_labels = data_pos
        g.ndata['h'] = self.gnn(g)
        g_out_n = mean_nodes(g, 'repr')
        bs, vt, hd = list(g_out_n.shape)
        g_out_n = g_out_n.view(bs, vt * hd)
        g_out = self.dropout(g_out_n)
        g_out = self.act(self.proj_ctx(g_out))

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        rel_embs = self.rel_emb(rel_labels)
        h_out = self.proj_ctx(head_embs.view(-1, self.params.num_layers * self.params.hidden_dim))
        t_out = self.proj_ctx(tail_embs.view(-1, self.params.num_layers * self.params.hidden_dim))
        head_embs = torch.nan_to_num_(head_embs, nan=0)
        tail_embs = torch.nan_to_num_(tail_embs, nan=0)

        ''' community view learning '''
        n_indices = g.ndata['idx'][head_ids]
        self.Hn.data[n_indices] = self.cluster_act(self.cluster_dropout(torch.spmm(h_out, self.Zn.t())))
        s1 = self.summary_act(self.dropout(torch.spmm(self.Hn.data[n_indices], self.Zn)))
        n_indices = g.ndata['idx'][tail_ids]
        self.Hn.data[n_indices] = self.cluster_act(self.cluster_dropout(torch.spmm(t_out, self.Zn.t())))
        s2 = self.summary_act(self.dropout(torch.spmm(self.Hn.data[n_indices], self.Zn)))
        s = s1 * s2

        ''' metapath view learning '''
        for hd, tl in zip(head_ids, tail_ids):
            edges_btw_roots = g.edge_id(hd, tl)
            tokens0.append(g.edata['token'][edges_btw_roots].unsqueeze(0))
            masks.append(g.edata['mask'][edges_btw_roots].unsqueeze(0))
            segments.append(g.edata['segment'][edges_btw_roots].unsqueeze(0))
        tokens = torch.cat(tokens0, dim=0).to(self.params.device)  # bs x ps x sq
        masks = torch.cat(masks, dim=0).to(self.params.device)  # bs x ps x sq
        segments = torch.cat(segments, dim=0).to(self.params.device)  # bs x ps x sq
        rep_seq, rep_rel = self.seq_model(rel_input_ids=tokens, rel_attention_mask=masks, rel_token_type_ids=segments)
        # print("positive :", tokens)

        ''' view attention-scores and aggregation '''
        view_embeds = torch.stack([g_out, rep_seq, s], dim=1)
        rel_embeds = rel_embs.unsqueeze(1)  # bs x 1 x hd
        view_embeds = torch.nan_to_num_(view_embeds, nan=0)
        rel_embeds = torch.nan_to_num_(rel_embeds, nan=0)
        self.attention_weights = torch.sum(rel_embeds * view_embeds, dim=-1)  # bs x view
        self.attention_weights = F.softmax(self.attention_weights, dim=-1)  # bs x view
        tmp_attention_weights = self.attention_weights.unsqueeze(-1)  # bs x view x 1
        view_aggr = torch.sum(tmp_attention_weights * view_embeds, dim=1)  # [bs, hd]

        ''' positive triplet scoring '''
        code1 = torch.empty(len(head_embs), 256)
        for i in range(len(head_ids)):
            code1[i] = torch.cat([self.one_hot_layer(return_one_hot(idx_id(head_ids[i].item()))), self.one_hot_layer(return_one_hot(idx_id(tail_ids[i].item())))], dim=0)
        g_rep = torch.cat([head_embs.view(-1, 512), tail_embs.view(-1, 512), rel_embs, view_aggr, code1], dim=1)
        out_pos = self.fc_layer(g_rep)



        ######################################## negative subgraph modeling ########################################
        ''' enclosing subgraph view learning '''
        g1, _, rel_labels1 = data_neg
        g1.ndata['h'] = self.gnn(g1)
        g_out1_n = mean_nodes(g1, 'repr')
        bs, vt, hd = list(g_out1_n.shape)
        g_out1_n = g_out1_n.view(bs, vt * hd)
        g_out1 = self.dropout(g_out1_n)
        g_out1 = self.act(self.proj_ctx(g_out1))

        head_ids1 = (g1.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs1 = g1.ndata['repr'][head_ids1]
        tail_ids1 = (g1.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs1 = g1.ndata['repr'][tail_ids1]
        rel_embs1 = self.rel_emb(rel_labels1)
        head_embs1 = torch.nan_to_num_(head_embs1, nan=0)
        tail_embs1 = torch.nan_to_num_(tail_embs1, nan=0)
        rel_embs1 = torch.nan_to_num_(rel_embs1, nan=0)

        hn_out = self.proj_ctx(head_embs1.view(-1, self.params.num_layers * self.params.hidden_dim))
        tn_out = self.proj_ctx(tail_embs1.view(-1, self.params.num_layers * self.params.hidden_dim))

        n_indices = g1.ndata['idx'][tail_ids1]
        self.Hn.data[n_indices] = self.cluster_act(self.cluster_dropout(torch.spmm(tn_out, self.Zn.t())))
        s3 = self.summary_act(self.dropout(torch.spmm(self.Hn.data[n_indices], self.Zn)))
        sn = s1 * s3
        if type in ["t", "v"]:
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg'][edges_btw_roots].unsqueeze(0))
        elif type == "p":
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg_p'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg_p'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg_p'][edges_btw_roots].unsqueeze(0))
        elif type == "r":
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg_r'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg_r'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg_r'][edges_btw_roots].unsqueeze(0))
        elif type == "2":
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg_2'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg_2'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg_2'][edges_btw_roots].unsqueeze(0))
        tokens = torch.cat(tokens0_neg, dim=0).to(self.params.device)  # bs x ps x sq
        masks = torch.cat(masks_neg, dim=0).to(self.params.device)  # bs x ps x sq
        segments = torch.cat(segments_neg, dim=0).to(self.params.device)  # bs x ps x sq
        rep_seq_neg, rep_rel_neg = self.seq_model(rel_input_ids=tokens, rel_attention_mask=masks, rel_token_type_ids=segments)

        view_embeds1 = torch.stack([g_out1, rep_seq_neg, sn], dim=1)  # TODO negative path-sampling
        view_embeds1 = torch.nan_to_num_(view_embeds1, nan=0)
        view_aggr1 = torch.sum(tmp_attention_weights * view_embeds1, dim=1)  # [bs, hd]
        code3 = torch.empty(len(head_embs1), 256)
        for i in range(len(head_embs1)):
           code3[i] = torch.cat([self.one_hot_layer(return_one_hot(idx_id(head_ids1[i].item()))), self.one_hot_layer(return_one_hot(idx_id(tail_ids1[i].item())))], dim=0)

        ''' negative triplet scoring '''
        g_rep1 = torch.cat([head_embs1.view(-1, 512), tail_embs1.view(-1, 512), rel_embs1, view_aggr1, code3], dim=1)
        out_neg = self.fc_layer(g_rep1)



        '''' modularity maximization based community detection '''
        a = torch.spmm(self.Hn.t(), torch.spmm(adj, self.Hn))
        ca = torch.spmm(self.Hn.t(), degrees)
        cb = torch.spmm(degrees.t(), self.Hn)
        normalizer = torch.spmm(ca, cb) / 2 / m
        spectral_loss = - torch.trace(a - normalizer) / 2 / m
        h1_l_loss.append(spectral_loss)
        pairwise = torch.spmm(self.Hn.t(), self.Hn)
        orthogonality_loss = torch.norm(pairwise / torch.norm(pairwise) - I / torch.sqrt(self.k))
        h1_o_loss.append(orthogonality_loss)
        # size_loss = torch.norm(pairwise.sum(1)) / dl.nodes['total'] * torch.sqrt(self.k) - 1
        # h1_s_loss.append(size_loss)

        ''' clustering loss computation '''
        h1_l_loss = self.params.cluster_learning_coeff * torch.stack(h1_l_loss).mean()
        h1_o_loss = self.params.cluster_orthogonality_coeff * torch.stack(h1_o_loss).mean()
        # h1_s_loss = self.params.cluster_size_coeff * torch.stack(h1_s_loss).mean()

        cluster_loss = h1_l_loss + h1_o_loss  # + h1_s_loss

        # 返回模型输出和评估指标
        return out_pos, out_neg, cluster_loss, logit, self.attention_weights, tokens0, view_embeds, view_aggr, rep_rel
        # return out_pos, out_neg, cluster_loss, logit, self.attention_weights, tokens0, view_embeds, view_aggr, rep_rel


class GraphBertForPairScoring(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphBertForPairScoring, self).__init__(config)
        self.config = config
        self.graphbert_path = GraphBert_path(config)
        self.epsilon = torch.FloatTensor([1e-20])
        self.proj_ctx = nn.Linear(config.hidden_size * config.path_samples * (config.num_hidden_layers + 1),
                                  config.hidden_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, src_input_ids=None, rel_input_ids=None, tgt_input_ids=None,
                src_attention_mask=None, tgt_attention_mask=None, rel_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None, rel_token_type_ids=None,
                src_embeds=None, rel_embeds=None, tgt_embeds=None,
                labels=None,
                label_dict=None, *args, **kwargs):
        bs, nt, sl = list(rel_input_ids.shape)  # 1024 5 7
        seq_rel, rep_rel, pooled_seq = self.encoder_path(  # bs*nt,hn
            rel_input_ids.view(bs * nt, -1),
            attention_mask=rel_attention_mask.view(bs * nt, -1),
            token_type_ids=rel_token_type_ids.view(bs * nt, -1), inputs_embeds=None, rel_flag=True)

        rep_rel = rep_rel.view(bs, nt * self.config.hidden_size)
        pooled_seq = pooled_seq.view(bs, nt * self.config.num_hidden_layers * self.config.hidden_size)
        all_rep = torch.cat([rep_rel, pooled_seq], dim=1)
        all_rep = self.dropout(all_rep)
        seq_embeddings = self.act(self.proj_ctx(all_rep))

        return seq_embeddings, rep_rel

    def encoder_path(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                     rel_flag=True):
        outputs = self.graphbert_path(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                      position_ids=position_ids, inputs_embeds=inputs_embeds, rel_flag=rel_flag)
        # print("outputs ", len(outputs)) 3
        # print("outputs[0] ", outputs[0].shape) # outputs[0]  torch.Size([5120, 7, 64])
        # print("outputs[1] ", outputs[1].shape) # outputs[1]  torch.Size([5120, 64])
        # print("outputs[2] ", len(outputs[2]))
        # print("outputs[2] ", outputs[2][0].shape) # torch.Size([5120, 7, 64])
        seq_output = outputs[0]
        pooled_output = outputs[1]
        pooled_seq = torch.stack(outputs[2], dim=1)
        return (seq_output, pooled_output, pooled_seq)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))

    def l2_norm(self, x, dim=1):
        return x / (torch.max(torch.norm(x, dim=dim, keepdim=True), self.epsilon))
