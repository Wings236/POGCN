import torch
from dataloader import Loader
from torch import nn
import world

class MFBPR(nn.Module):
    def __init__(self, config:dict, dataset:Loader):
        super(MFBPR, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 
        nn.init.normal_(self.embedding_user.weight)
        nn.init.normal_(self.embedding_item.weight)
        print("using Normal distribution N(0,1) initialization for PO-BPR")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg, flag):
        
        pos_emb   = self.embedding_item(pos.long())
        users_emb = self.embedding_user(users.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        if flag == 0:
            loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        else:
            loss = 0.1*torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        
        
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss


class POGCN(nn.Module):
    def __init__(self, config:dict, dataset:Loader):
        super(POGCN, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']
        self.dropout = config["dropout"]
        self.keep_prob = config["keep_prob"]
        # self.Graph = dataset.behavior_adj.to(world.device)
        self.PO_Graph = dataset.PO_behavior_adj.to(world.device)
        self.behavior_index_map = dataset.behavior_index_map
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print(f"POGcn is already to go")

    def __dropout(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __po_dropout(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        length = len(self.behavior_index_map)
        index_list, values_list = [], []
        for idx, temp_map in enumerate(self.behavior_index_map):
            temp_keep =  keep_prob +(1-keep_prob)/length*(length-idx-1)
            random_index = torch.rand(len(temp_map)) + temp_keep
            random_index = random_index.int().bool()
            index_list.append(index[temp_map][random_index])
            values_list.append(values[temp_map][random_index]/temp_keep)
        index = torch.concat(index_list, dim=0)
        values = torch.concat(values_list, dim=0)

        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
        
    
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        Graph = self.PO_Graph
        # Graph = self.Graph

        # dropout
        # print(self.dropout)
        # print(self.training)
        if self.dropout:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(Graph, self.keep_prob)
                # g_droped = self.__po_dropout(Graph, self.keep_prob)
            else:
                g_droped = Graph       
        else:
            g_droped = Graph

        # propagation
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def pre_score(self):
        self.training = False
        users_emb, items_emb = self.computer()
        user_batch_size = 100
        user_batch_num = users_emb.shape[0] // user_batch_size + 1
        scores = []
        for i in range(user_batch_num): scores.append(self.f(torch.matmul(users_emb[i*user_batch_size:(i+1)*user_batch_size], items_emb.t()).cpu()))
        self.score = torch.cat(scores, dim=0)


    def getUsersRating(self, users):
        return self.score[users]

    def bpr_loss(self, users, pos, neg, flag=0):
        users_emb, items_emb = self.computer()
        
        user_emb = users_emb[users.long()]
        pos_emb = items_emb[pos.long()]
        neg_emb = items_emb[neg.long()]
        reg_loss = (1/2)*(user_emb.norm(2).pow(2) + 
                         pos_emb.norm(2).pow(2)  +
                         neg_emb.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(user_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)


        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
    