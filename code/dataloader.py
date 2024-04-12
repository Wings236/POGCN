import numpy as np
import pandas as pd
import pickle
from os.path import join
from itertools import combinations
import scipy.sparse as sp
from torch.utils.data import Dataset
from time import time
from world import cprint
import torch
import world
import os

class Loader(Dataset):
    def __init__(self, root_path="../data/taobao", behavior_list=["click","buy"]):
        cprint(f'loading [{root_path}]')
        total_user_item_num_path = join(root_path, "ui_num.pkl")
        with open(total_user_item_num_path, "rb") as f:
            user_item_dict = pickle.load(f)
            self.n_user = user_item_dict["user_num"]
            self.m_item = user_item_dict["item_num"]
        cprint(f'dataset INFO:')
        print(f"user num:{self.n_user}, item num:{self.m_item}")
        
        self.old_behavior_list = behavior_list
        behavior_list = []
        for order_bh in self.old_behavior_list:
            if isinstance(order_bh, list):
                for sub_bh in order_bh:
                    behavior_list.append(sub_bh)
            else:
                behavior_list.append(order_bh)
        dataset_list = []
        for b in behavior_list:
            path = join(root_path, b)
            train_file = path + '/train.txt'
            valid_file = path + '/valid.txt'
            test_file = path + '/test.txt'
            trainItem, trainUser = [], []
            validItem, validUser = [], []
            testItem, testUser = [], []

            with open(train_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        if len(items) == 0:
                            continue
                        uid = int(l[0])
                        trainUser.extend([uid] * len(items))
                        trainItem.extend(items)
            df_train = pd.DataFrame({"user_id":trainUser, "item_id":trainItem})
            with open(valid_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        if len(items) == 0:
                            continue
                        uid = int(l[0])
                        validUser.extend([uid] * len(items))
                        validItem.extend(items)
            df_valid = pd.DataFrame({"user_id":validUser, "item_id":validItem})
            with open(test_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        if len(items) == 0:
                            continue
                        uid = int(l[0])
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
            df_test = pd.DataFrame({"user_id":testUser, "item_id":testItem})
            dataset_list.append((df_train, df_valid, df_test))
            # Output the behavior data
            cprint(b)
            print(f"behavior interations {df_train.shape[0] + df_valid.shape[0] + df_test.shape[0]}")
            print(f"{df_train.shape[0]} interactions for training")
            print(f"{df_valid.shape[0]} interactions for validing")
            print(f"{df_test.shape[0]} interactions for testing")
        
        df_train_all = pd.DataFrame(columns=["user_id","item_id"])
        df_valid_all = pd.DataFrame(columns=["user_id","item_id"])
        df_test_all = pd.DataFrame(columns=["user_id","item_id"])
        for df_train, df_valid, df_test in dataset_list:
            df_train_all = pd.concat([df_train_all, df_train], axis=0)
            df_valid_all = pd.concat([df_valid_all, df_valid], axis=0)
            df_test_all = pd.concat([df_test_all, df_test], axis=0)
        
        # all train dataset
        self.train_all = df_train_all.drop_duplicates()
        self.valid_all = df_valid_all.drop_duplicates()
        self.test_all = df_test_all.drop_duplicates()
        cprint("All data train")
        print(f"all interations {self.train_all.shape[0] + self.valid_all.shape[0] + self.test_all.shape[0]}")
        print(f"{self.train_all.shape[0]} interactions for training")
        print(f"{self.valid_all.shape[0]} interactions for validing")
        print(f"{self.test_all.shape[0]} interactions for testing")
        
        # pre-calculate
        self.dataset_list = dataset_list
        self.behavior_list = behavior_list

        self.partial_order_dataset = self.partial_order_relation()
        self.po_Pos, self.po_num = self.build_partial_order_train()

        self.allPos, self.allvalidDict, self.alltestDict = self.build_all_train_valid_test()
        self.bh_Pos, self.bh_validDict, self.bh_testDict = self.build_behavior_train_valid_test()
        # adj_path = join(root_path, "POGCN_data.pkl")
        # if os.path.exists(adj_path):
        #     print("Loading the pre-computer")
        #     with open(adj_path, "rb") as f:
        #         Graph_dict = pickle.load(f)
        #         self.behavior_adj = Graph_dict["norm"]
        #         # self.PO_behavior_adj = Graph_dict["Po_norm"]
        # else:
        #     print("pre-calculate the adjacent matrix with lapalacian normalization")
        #     s = time()
        #     self.behavior_adj = self.build_behavior_UI_adjacent_matrix_with_norm()
        #     # self.PO_behavior_adj = self.build_behavior_UI_adjacent_matrix_with_Po_norm()
        #     Graph_dict = {"norm":self.behavior_adj}
        #     with open(adj_path, "wb") as f:
        #         pickle.dump(Graph_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        #     print(f"time cost {time()-s:.2f}s")

        self.PO_behavior_adj, self.behavior_index_map = self.build_behavior_UI_adjacent_matrix_with_Po_norm()

        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item

    def build_all_train_valid_test(self):
        train_pos_items, valid_pos_items, test_pos_items = dict(), dict(), dict()
        for user_id, g in self.train_all.groupby("user_id"):
            train_pos_items[user_id] = g["item_id"].to_list()
        for user_id, g in self.valid_all.groupby("user_id"):
            valid_pos_items[user_id] = g["item_id"].to_list()
        for user_id, g in self.test_all.groupby("user_id"):
            test_pos_items[user_id] = g["item_id"].to_list()
        return train_pos_items, valid_pos_items, test_pos_items

    def build_behavior_train_valid_test(self):
        train_data = [dict() for _ in self.behavior_list]
        valid_data = [dict() for _ in self.behavior_list]
        test_data = [dict() for _ in self.behavior_list]
        for i in range(len(train_data)):
            temp_train = self.dataset_list[i][0]
            temp_valid = self.dataset_list[i][1]
            temp_test = self.dataset_list[i][2]
            for user_id, g in temp_train.groupby("user_id"):
                train_data[i][user_id] = g["item_id"].to_list()
            for user_id, g in temp_valid.groupby("user_id"):
                valid_data[i][user_id] = g["item_id"].to_list()
            for user_id, g in temp_test.groupby("user_id"):
                test_data[i][user_id] = g["item_id"].to_list()
        return train_data, valid_data, test_data

    def _build_sparse_matrix(self, df, name1, name2, shape1, shape2):
        R = sp.csr_matrix((np.ones(len(df)), (df[name1].to_numpy(), df[name2].to_numpy())), shape=(shape1, shape2))
        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        coo = norm_adj.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()

    def build_behavior_UI_adjacent_matrix_with_norm(self):
        return self._build_sparse_matrix(self.train_all, "user_id", "item_id", self.n_users, self.m_items)
    
    def partial_order_relation(self):
        res = [self.train_all]
        bh_name = self.old_behavior_list[1:][::-1]
        dataset_list = self.dataset_list[1:][::-1]
        offsetidx = 0
        for idx, bh in enumerate(bh_name):
            temp = []
            for temp_df in res:
                if isinstance(bh, list):
                    temp_list = [[] for _ in range(len(bh)+1)]
                    temp_list[0] = [temp_df]
                    for bh_idx in range(len(bh)):
                        temp_train_df = dataset_list[idx+offsetidx+bh_idx][0]
                        temp_res_list = [[] for _ in range(len(bh)+1)]
                        for list_idx, df_list in enumerate(temp_list):
                            if len(df_list) == 0:
                                continue
                            else:
                                for temp_temp_df in df_list:
                                    inner_df = pd.merge(temp_temp_df, temp_train_df)
                                    temp_res_list[list_idx+1].append(inner_df)
                                    diff_df = pd.concat([temp_temp_df, temp_train_df], ignore_index=True).drop_duplicates(keep=False)
                                    temp_res_list[list_idx].append(diff_df)
                        temp_list = temp_res_list
                    for temp_df_list in temp_list:
                        temp.append(pd.concat(temp_df_list, ignore_index=True).drop_duplicates(keep=False))
                else:
                    train_df = dataset_list[idx+offsetidx][0]
                    inner_df = pd.merge(temp_df, train_df)
                    temp.append(inner_df)
                    diff_df = pd.concat([temp_df, inner_df], ignore_index=True).drop_duplicates(keep=False)
                    temp.append(diff_df)
                    
            res = temp
            if isinstance(bh, list): offsetidx += len(bh) - 1 
                    
        return res
    
    def build_partial_order_train(self):
        train_data = [dict() for _ in self.partial_order_dataset]
        train_num = []
        for i in range(len(train_data)):
            temp_train = self.partial_order_dataset[i]
            train_num.append(len(temp_train))
            for user_id, g in temp_train.groupby("user_id"):
                train_data[i][user_id] = g["item_id"].to_list()
        return train_data, train_num
    
    def _build_partial_order_sparse_matrix(self, df, name1, name2, shape1, shape2):
        behavior_R = sum([sp.csr_matrix((np.ones(len(train_df))*(len(df)-idx)**world.config['level'], (train_df[name1].to_numpy(), train_df[name2].to_numpy())), shape=(shape1, shape2)) for idx, train_df in enumerate(df)])
        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:self.n_users, self.n_users:] = behavior_R
        adj_mat[self.n_users:, :self.n_users] = behavior_R.T

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)

        coo = norm_adj.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        adj_mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()

        behavior_index_map = []
        b = adj_mat.indices().t()
        hash_table = {}
        for i in range(b.shape[0]):
            hash_table[tuple(b[i].tolist())] = i
        
        for train_df in df:
            a = torch.tensor(train_df.to_numpy(np.int32))
            a[:,1] = a[:,1] + self.n_user
            a = torch.cat([a, a.flip(dims=[1])])
            
            indices = []
            for i in range(a.shape[0]):
                key = tuple(a[i].tolist())
                indices.append(hash_table[key])

            behavior_index_map.append(indices)

        return adj_mat, behavior_index_map

    def build_behavior_UI_adjacent_matrix_with_Po_norm(self):
        return self._build_partial_order_sparse_matrix(self.partial_order_dataset, "user_id", "item_id", self.n_users, self.m_items)