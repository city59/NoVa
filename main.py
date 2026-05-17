import datetime
import gc
import os
import pickle
import random
import time

import numpy as np
import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
from numpy import random
from tqdm import tqdm

import BGNN
import DataHandler
import MV_Net
import graph_utils
from HNT import LAGCLLinkPredictionModel
from Params import args
from Utils.TimeLogger import log
from graph_builder import build_graph
from lagcl_encoder import Discriminator

t.backends.cudnn.benchmark=True

if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

MAX_FLAG = 0x7FFFFFFF

now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

t.autograd.set_detect_anomaly(True)

class Model():
    def __init__(self):
   

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'     
        # self.tst_file = args.path + args.dataset + '/BST_tst_int_59' 
        #Tmall: 3,4,5,6,8,59
        #IJCAI_15: 5,6,8,10,13,53

        # self.meta_multi_file = args.path + args.dataset + '/meta_multi_beh_user_index'
        # self.meta_single_file = args.path + args.dataset + '/meta_single_beh_user_index'
        self.meta_multi_single_file = args.path + args.dataset + '/meta_multi_single_beh_user_index_shuffle'
        #                                                         /meta_multi_single_beh_user_index_shuffle
        #                                                         /new_multi_single

        # self.meta_multi = pickle.load(open(self.meta_multi_file, 'rb'))
        # self.meta_single = pickle.load(open(self.meta_single_file, 'rb'))
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb'))

        self.t_max = -1 
        self.t_min = 0x7FFFFFFF
        self.time_number = -1
 
        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {} 
        self.behaviors = []
        self.behaviors_data = {}
     
        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()  #

        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0

            
        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv','fav', 'cart', 'buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click','fav', 'cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['click','fav', 'cart', 'buy']

        elif args.dataset == 'JD':
            self.behaviors = ['review','browse', 'buy']
            self.behaviors_SSL = ['review','browse', 'buy']

        elif args.dataset == 'retailrocket':
            self.behaviors = ['view','cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['view','cart', 'buy']


        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:  
                data = pickle.load(fs)
                self.behaviors_data[i] = data 

                if data.get_shape()[0] > self.user_num:  
                    self.user_num = data.get_shape()[0]  
                if data.get_shape()[1] > self.item_num:  
                    self.item_num = data.get_shape()[1]

             
                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

        
                if self.behaviors[i]==args.target:
                    self.trainMat = data
                    self.trainLabel = 1*(self.trainMat != 0)  
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))  


        time = datetime.datetime.now()
        print("Start building:  ", time)
        for i in range(0, len(self.behaviors)):
            self.behavior_mats[i] = graph_utils.get_use(self.behaviors_data[i])                  
        time = datetime.datetime.now()
        print("End building:", time)


        print("user_num: ", self.user_num)
        print("item_num: ", self.item_num)
        print("\n")


        #---------------------------------------------------------------------------------------------->>>>>
        #train_data
        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

        #valid_data


        # test_data  
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        # tstUsrs = np.reshape(np.argwhere(data!=None), [-1])
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist()
        # 筛选出在训练集中出现过的用户、物品(lagcl模型测试时要满足的条件)
        test_data = [i for i in test_data if i[0] in train_u and i[1] in train_v]
        # testbatch = np.maximum(1, args.batch * args.sampNum 
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)  
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)  
        # -------------------------------------------------------------------------------------------------->>>>>

        # --------------------LAGCL数据预处理-------------------------
        # 加载数据集和构建图
        # 调用build_graph函数加载数据并构建二部图，返回图对象和相关数据结构
        self.graph, self.node_table, train_edges, (self.training_set_u, self.training_set_i, self.test_set), (self.node_table_dict_by_node_id, node_table_dict_by_node_idx) = build_graph(train_data, test_data)

        # 读取训练样本并准备映射字典
        self.training_data = train_edges[['node1_id', 'node2_id']].values.tolist()  # 提取训练边的用户-物品对
        # 节点ID到特征索引的映射，用于在嵌入层中查找节点特征
        self.node_raw_idx_mapping = {
            key: value['node_feature']  # 节点ID到特征索引的映射
            for key, value in self.node_table_dict_by_node_id.items()
        }
        # 物品索引到物品ID的映射，用于推荐结果的转换和展示
        self.id2item = {
            value['node_feature'] - self.user_num: key  # 物品特征索引减去用户数量得到物品索引
            for key, value in self.node_table_dict_by_node_id.items()
            if value['node_type'] == 1  # 只选择物品节点（类型为1）
        }

    def prepareModel(self):
        self.modelName = self.getModelName()  
        # self.setRandomSeed()
        self.gnn_layer = eval(args.gnn_layer)  
        self.hidden_dim = args.hidden_dim
        

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        else:
            self.model = BGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
            self.meta_weight_net = MV_Net.MetaWeightNet(len(self.behaviors)).cuda()
                    


        # #IJCAI_15
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=2, step_size_down=3, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        # #       


        #Tmall
        self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=3, step_size_down=7, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        #                                                                                                                                                                           0.993                                             

        # # retailrocket
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # self.meta_opt =  t.optim.SGD(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, nesterov=True)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=1, step_size_down=2, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=1, step_size_down=2, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        # #                                                                                                                                      exp_range


        if use_cuda:
            self.model = self.model.cuda()

        # ------LAGCL模型定义-------
        # 初始化LAGCL链接预测模型
        self.lagcl_model = LAGCLLinkPredictionModel(
            graph=self.graph,  # 输入的二部图对象
            user_num=self.user_num,  # 用户节点数量
            item_num=self.item_num,  # 物品节点数量
            hidden_dim=args.hidden_size,  # 隐藏层维度，决定嵌入向量的大小
            n_hops=args.n_hops,  # GNN的层数，即消息传递的跳数
            noise_eps=args.noise_eps,  # 噪声扰动幅度，用于对比学习中的数据增强
            cl_tau=args.cl_tau,  # 对比学习温度参数，控制正负样本的区分度
            cl_rate=args.cl_rate,  # 对比学习损失权重，平衡主任务和对比学习任务
            tail_k_threshold=args.tail_k_threshold,  # 长尾阈值，区分头部和尾部节点
            hesitation_threshold=args.hesitation_threshold, # 犹豫集阈值
            hesitation_weight=args.hesitation_weight, # 犹豫集损失权重
            m_h_eps=args.m_h_eps,  # 知识迁移模块损失权重，控制从头部到尾部的知识迁移
            kl_eps=args.kl_eps,  # KL散度损失权重，确保增强后的表示与原始表示相似
            l2_reg=args.l2_reg,  # L2正则化系数，防止过拟合
            neighbor_norm_type=args.neighbor_norm_type,  # 邻居归一化类型（left/right/both）
            agg_function=args.agg_function,  # 聚合函数类型，决定如何聚合邻居信息
            annealing_type=args.annealing_type,  # 退火类型，控制训练过程中参数的变化
            lambda_gp=args.lambda_gp,  # 梯度惩罚系数，用于对抗训练
            use_relation_rf=args.use_relation_rf,  # 是否使用关系预测模块
            agg_w_exp_scale=args.agg_w_exp_scale,  # 聚合权重指数缩放，调整邻居重要性
            device='cuda',  # 计算设备（GPU或CPU）
        )

        # 初始化优化
        # 使用Adam优化器，适合处理稀疏梯度和大规模数据
        self.optimizer = t.optim.Adam(self.lagcl_model.parameters(), lr=args.lagcl_lr)
        # 初始化判别器模型
        # 判别器用于对抗训练，区分真实样本和生成样本
        self.disc_model = Discriminator(args.hidden_size).cuda()  # 主判别器
        self.disc_pseudo_real = Discriminator(args.hidden_size).cuda()  # 伪真实判别器
        # 初始化判别器优化器
        # 使用RMSprop优化器，适合对抗训练场景
        self.optimizer_D = t.optim.RMSprop(self.disc_model.parameters(), lr=args.lagcl_lr * 0.1)  # 判别器学习率通常小于生成器
        self.optimizer_D_pseudo_real = t.optim.RMSprop(self.disc_pseudo_real.parameters(), lr=args.lagcl_lr * 0.1)

    def innerProduct(self, u, i, j):  
        pred_i = t.sum(t.mul(u,i), dim=1)*args.inner_product_mult  
        pred_j = t.sum(t.mul(u,j), dim=1)*args.inner_product_mult
        return pred_i, pred_j

    def SSL(self, user_embeddings, item_embeddings, target_user_embeddings, target_item_embeddings, user_step_index):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]  
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,t.randperm(corrupted_embedding.size()[1])]  
            return corrupted_embedding
        def score(x1, x2):
            return t.sum(t.mul(x1, x2), 1)

        def neg_sample_pair(x1, x2, τ = 0.05):  
            for i in range(x1.shape[0]):
                index_set = set(np.arange(x1.shape[0]))
                index_set.remove(i)
                index_set_neg = t.as_tensor(np.array(list(index_set))).long().cuda()  

                x_pos = x1[i].repeat(x1.shape[0]-1, 1)
                x_neg = x2[index_set]  
                
                if i==0:
                    x_pos_all = x_pos
                    x_neg_all = x_neg
                else:
                    x_pos_all = t.cat((x_pos_all, x_pos), 0)
                    x_neg_all = t.cat((x_neg_all, x_neg), 0)
            x_pos_all = t.as_tensor(x_pos_all)  #[9900, 100]
            x_neg_all = t.as_tensor(x_neg_all)  #[9900, 100]  

            return x_pos_all, x_neg_all

        def one_neg_sample_pair_index(i, step_index, embedding1, embedding2):

            index_set = set(np.array(step_index))
            index_set.remove(i.item())
            neg2_index = t.as_tensor(np.array(list(index_set))).long().cuda()

            neg1_index = t.ones((2,), dtype=t.long)
            neg1_index = neg1_index.new_full((len(index_set),), i)

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze())
            return neg_score_pre

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2):  #small, big, target, beh: [100], [1024], [31882, 16], [31882, 16]

            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set                         #beh
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().cuda()  #[910]
            neg2_index = t.unsqueeze(neg2_index, 0)                              #[1, 910]
            neg2_index = neg2_index.repeat(len(batch_index), 1)                  #[100, 910]
            neg2_index = t.reshape(neg2_index, (1, -1))                          #[1, 91000]
            neg2_index = t.squeeze(neg2_index)                                   #[91000]
                                                                                 #target
            neg1_index = batch_index.long().cuda()     #[100]
            neg1_index = t.unsqueeze(neg1_index, 1)                              #[100, 1]
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))               #[100, 910]
            neg1_index = t.reshape(neg1_index, (1, -1))                          #[1, 91000]           
            neg1_index = t.squeeze(neg1_index)                                   #[91000]

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)  #[91000,1]==>[91000]==>[100, 910]==>[100]
            return neg_score_pre  #[100]

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ = 0.05):  #[1024, 16], [1024, 16]

            if neg1_index!=None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]  
            D = x1.shape[1]

            x1 = x1
            x2 = x2

            scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1)+1e-8))  #[1024, 1]
            
            return scores
        def single_infoNCE_loss_simple(embedding1, embedding2):
            pos = score(embedding1, embedding2)  #[100]
            neg1 = score(embedding2, row_column_shuffle(embedding1))  
            one = t.cuda.FloatTensor(neg1.shape[0]).fill_(1)  #[100]
            # one = zeros = t.ones(neg1.shape[0])
            con_loss = t.sum(-t.log(1e-8 + t.sigmoid(pos))-t.log(1e-8 + (one - t.sigmoid(neg1))))  
            return con_loss

        #use_less    
        def single_infoNCE_loss(embedding1, embedding2):
            N = embedding1.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1, embedding2).squeeze()  #[100, 1]

            neg_x1, neg_x2 = neg_sample_pair(embedding1, embedding2)  #[9900, 100], [9900, 100]
            neg_score = t.sum(compute(neg_x1, neg_x2).view(N, (N-1)), dim=1)  #[100]  
            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score))   
            con_loss = t.mean(con_loss)  
            return max(0, con_loss)

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):  #target, beh
            N = step_index.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()  #[1024]
            neg_score = t.zeros((N,), dtype = t.float64).cuda()  #[1024]

            #-------------------------------------------------multi version-----------------------------------------------------
            steps = int(np.ceil(N / args.SSL_batch))  #separate the batch to smaller one 
            for i in range(steps):
                st = i * args.SSL_batch
                ed = min((i+1) * args.SSL_batch, N)
                batch_index = step_index[st: ed]

                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2)
                if i ==0:
                    neg_score = neg_score_pre
                else:
                    neg_score = t.cat((neg_score, neg_score_pre), 0)
            #-------------------------------------------------multi version-----------------------------------------------------

            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score+1e-8))  #[1024]/[1024]==>1024


            assert not t.any(t.isnan(con_loss))
            assert not t.any(t.isinf(con_loss))

            return t.where(t.isnan(con_loss), t.full_like(con_loss, 0+1e-8), con_loss)

        user_con_loss_list = []
        item_con_loss_list = []

        SSL_len = int(user_step_index.shape[0]/10)
        user_step_index = t.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()

        for i in range(len(self.behaviors_SSL)):

            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))

        user_con_losss = t.stack(user_con_loss_list, dim=0)  

        return user_con_loss_list, user_step_index  #4*[1024]

    def run(self):
     
        self.prepareModel()
        if args.isload == True:
            print("----------------------pre test:")
            HR, NDCG, LAGCL_RECALL_20, LAGCL_NDCG_20 = self.testEpoch(self.test_loader)
            print(f"HR: {HR} , NDCG: {NDCG}, LAGCL_RECALL_20: {LAGCL_RECALL_20}, LAGCL_NDCG_20: {LAGCL_NDCG_20}")
        log('Model Prepared')


        cvWait = 0  
        self.best_HR = 0 
        self.best_NDCG = 0
        flag = 0

        self.user_embed = None 
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None


        print("Test before train:")
        HR, NDCG, LAGCL_RECALL_20, LAGCL_NDCG_20 = self.testEpoch(self.test_loader)

        for e in range(self.curEpoch, args.epoch+1):  
            self.curEpoch = e

            self.meta_flag = 0
            if e%args.meta_slot == 0:
                self.meta_flag=1


            log("*****************Start epoch: %d ************************"%e)  

            if args.isJustTest == False:
                epoch_loss, user_embed, item_embed, user_embeds, item_embeds = self.trainEpoch()
                self.train_loss.append(epoch_loss)  
                print(f"epoch {e/args.epoch},  epoch loss{epoch_loss}")
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG, LAGCL_RECALL_20, LAGCL_NDCG_20 = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.scheduler.step()
            self.meta_scheduler.step()

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch 
                cvWait = 0
                print("--------------------------------------------------------------------------------------------------------------------------best_HR", self.best_HR)
                # print("--------------------------------------------------------------------------------------------------------------------------NDCG", self.best_NDCG)
                self.user_embed = user_embed 
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()


            
            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch 
                cvWait = 0
                # print("--------------------------------------------------------------------------------------------------------------------------HR", self.best_HR)
                print("--------------------------------------------------------------------------------------------------------------------------best_NDCG", self.best_NDCG)
                self.user_embed = user_embed 
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()



            if (HR<self.best_HR) and (NDCG<self.best_NDCG): 
                cvWait += 1


            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                self.saveHistory()
                self.saveModel()
                break
               
        HR, NDCG, LAGCL_RECALL, LAGCL_NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)

    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = [] 
        item_id_pos = [] 
        item_id_neg = [] 
 
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(args.sampNum, len(posset))   
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item()) 
                item_id_neg.append(neglocs[j])
                cur += 1

        return t.as_tensor(np.array(user_id), dtype=t.long).cuda(), t.as_tensor(np.array(item_id_pos), dtype=t.long).cuda(), t.as_tensor(np.array(item_id_neg),dtype=t.long).cuda()

    def this_batch_pairwise(self, training_user, training_item):
        """
        生成训练批次数据的生成器函数，采用成对采样方式

        该函数实现了推荐系统中常用的成对（Pairwise）采样策略，为BPR损失函数准备训练数据：
        1. 为每个正样本随机采样负样本（用户未交互过的物品）
        2. 将用户、正样本物品和负样本物品的特征索引组织成三元组返回

        参数:
            training_data (list): 训练数据列表，每项包含用户ID和物品ID对

        返回:
            generator: 生成(u_idx, i_idx, j_idx)三元组，分别是用户索引、正样本物品索引和负样本物品索引
        """
        from random import choice  
        n_negs = 1 

      
        users = training_user
     
        items = training_item

        u_idx, i_idx, j_idx = [], [], [] 
        item_list = list(self.training_set_i.keys()) 

        for i, user_id in enumerate(users):
            i_idx.append(self.node_table_dict_by_node_id[items[i]]['node_feature']) 
            u_idx.append(self.node_table_dict_by_node_id[user_id]['node_feature']) 

            for _ in range(n_negs):
                neg_item_id = choice(item_list)  
                while neg_item_id in self.training_set_u[user_id]:
                    neg_item_id = choice(item_list)
                j_idx.append(self.node_table_dict_by_node_id[neg_item_id]['node_feature']) 
        return u_idx, i_idx, j_idx 

    def trainEpoch(self):   
        train_loader = self.train_loader
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)
        
        epoch_loss = 0
    
#-----------------------------------------------------------------------------------
        self.behavior_loss_list = [None]*len(self.behaviors)      

        self.user_id_list = [None]*len(self.behaviors)
        self.item_id_pos_list = [None]*len(self.behaviors)
        self.item_id_neg_list = [None]*len(self.behaviors)

        self.meta_start_index = 0
        self.meta_end_index = self.meta_start_index + args.meta_batch
#----------------------------------------------------------------------------------
        # LAGCL模型训练前处理
        self.lagcl_model.cuda() 
        self.lagcl_model.train() 

       
        iter_num = len(self.train_loader) 
        discD_every_iter = iter_num // 16  
        discG_every_iter = discD_every_iter * 4  

        lagcl_total_loss = 0  

#----------------------------------------------------------------------------------

        cnt = 0
        for user, item, item_i, item_j in tqdm(train_loader):

            user = user.long().cuda()
            item = item.long().cuda()
            self.user_step_index = user


            self.meta_user = t.as_tensor(self.meta_multi_single[self.meta_start_index:self.meta_end_index]).cuda()  
            
            if self.meta_end_index == self.meta_multi_single.shape[0]:
                self.meta_start_index = 0  
            else:
                self.meta_start_index = (self.meta_start_index + args.meta_batch) % (self.meta_multi_single.shape[0] - 1)
            self.meta_end_index = min(self.meta_start_index + args.meta_batch, self.meta_multi_single.shape[0])

# ---round zero---------------------------------------------------------------------------------------------
            # 转换数据格式为LGACL符合的数据
            train_user = [f'userid_{id}' for id in user.tolist()]
            train_item = [f'itemid_{id}' for id in item.tolist()]
            u_idx, i_idx, j_idx = self.this_batch_pairwise(train_user, train_item)
            # LAGCL模型训练
            self.optimizer.zero_grad()  
            _, loss = self.lagcl_model((u_idx, i_idx, j_idx))  
            lagcl_total_loss += loss.item()  
            loss.backward()  
            self.optimizer.step() 

           
            if cnt % discD_every_iter == 0: 
                self.lagcl_model.train_disc(self.lagcl_model, self.disc_model, self.optimizer_D, self.disc_pseudo_real, self.optimizer_D_pseudo_real, cnt, iter_num)
                
                if cnt % discG_every_iter == 0 and cnt > 0:  
                    self.lagcl_model.train_gen(self.lagcl_model, self.optimizer, self.disc_model, self.disc_pseudo_real)


# ---round zero---------------------------------------------------------------------------------------------


#---round one---------------------------------------------------------------------------------------------

            meta_behavior_loss_list = [None]*len(self.behaviors)
            meta_user_index_list = [None]*len(self.behaviors)  #---

            meta_model = BGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
            meta_opt = t.optim.AdamW(meta_model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
            meta_model.load_state_dict(self.model.state_dict())

            meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds = meta_model()


            for index in range(len(self.behaviors)):

                not_zero_index = np.where(item_i[index].cpu().numpy()!=-1)[0]

                self.user_id_list[index] = user[not_zero_index].long().cuda()
                meta_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                meta_userEmbed = meta_user_embed[self.user_id_list[index]]
                meta_posEmbed = meta_item_embed[self.item_id_pos_list[index]]
                meta_negEmbed = meta_item_embed[self.item_id_neg_list[index]]

                meta_pred_i, meta_pred_j = 0, 0
                meta_pred_i, meta_pred_j = self.innerProduct(meta_userEmbed, meta_posEmbed, meta_negEmbed)
                
                meta_behavior_loss_list[index] = - (meta_pred_i.view(-1) - meta_pred_j.view(-1)).sigmoid().log()


            meta_infoNCELoss_list, SSL_user_step_index = self.SSL(meta_user_embeds, meta_item_embeds, meta_user_embed, meta_item_embed, self.user_step_index)

            meta_infoNCELoss_list_weights, meta_behavior_loss_list_weights = self.meta_weight_net(\
                                                                         meta_infoNCELoss_list, \
                                                                         meta_behavior_loss_list, \
                                                                         SSL_user_step_index, \
                                                                         meta_user_index_list, \
                                                                         meta_user_embeds, \
                                                                         meta_user_embed)



            for i in range(len(self.behaviors)):
                meta_infoNCELoss_list[i] = (meta_infoNCELoss_list[i]*meta_infoNCELoss_list_weights[i]).sum()
                meta_behavior_loss_list[i] = (meta_behavior_loss_list[i]*meta_behavior_loss_list_weights[i]).sum()   


            meta_bprloss = sum(meta_behavior_loss_list) / len(meta_behavior_loss_list)
            meta_infoNCELoss = sum(meta_infoNCELoss_list) / len(meta_infoNCELoss_list)
            meta_regLoss = (t.norm(meta_userEmbed) ** 2 + t.norm(meta_posEmbed) ** 2 + t.norm(meta_negEmbed) ** 2)            

            meta_model_loss = (meta_bprloss + args.reg * meta_regLoss + args.beta*meta_infoNCELoss) / args.batch

            meta_opt.zero_grad(set_to_none=True)
            self.meta_opt.zero_grad(set_to_none=True)
            meta_model_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=20, norm_type=2)
            meta_opt.step()
            self.meta_opt.step()
   
#---round one---------------------------------------------------------------------------------------------



#---round two---------------------------------------------------------------------------------------------

            behavior_loss_list = [None]*len(self.behaviors)
            user_index_list = [None]*len(self.behaviors)  #---

            user_embed, item_embed, user_embeds, item_embeds = meta_model()

            for index in range(len(self.behaviors)):

                user_id, item_id_pos, item_id_neg = self.sampleTrainBatch(t.as_tensor(self.meta_user), self.behaviors_data[index])

                user_index_list[index] = user_id


                userEmbed = user_embed[user_id]

                posEmbed = item_embed[item_id_pos]
                negEmbed = item_embed[item_id_neg]

                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)
                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()  
              


            self.infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, item_embeds, user_embed, item_embed, self.meta_user)

            infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(\
                                                                         self.infoNCELoss_list, \
                                                                         behavior_loss_list, \
                                                                         SSL_user_step_index, \
                                                                         user_index_list, \
                                                                         user_embeds, \
                                                                         user_embed)


            for i in range(len(self.behaviors)):
                self.infoNCELoss_list[i] = (self.infoNCELoss_list[i]*infoNCELoss_list_weights[i]).sum()
                behavior_loss_list[i] = (behavior_loss_list[i]*behavior_loss_list_weights[i]).sum()   

            bprloss = sum(behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(self.infoNCELoss_list) / len(self.infoNCELoss_list)
            round_two_regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)


            meta_loss = 0.5 * (bprloss + args.reg * round_two_regLoss  + args.beta*infoNCELoss) / args.batch

            self.meta_opt.zero_grad()
            meta_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            self.meta_opt.step()

#---round two-----------------------------------------------------------------------------------------------



#---round three---------------------------------------------------------------------------------------------
 

            user_embed, item_embed, user_embeds, item_embeds = self.model()


            for index in range(len(self.behaviors)):


                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]

                pred_i, pred_j = 0, 0
                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()  

            infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, item_embeds, user_embed, item_embed, self.user_step_index)

            with t.no_grad():
                infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(\
                                                                            infoNCELoss_list, \
                                                                            self.behavior_loss_list, \
                                                                            SSL_user_step_index, \
                                                                            self.user_id_list, \
                                                                            user_embeds, \
                                                                            user_embed)


            for i in range(len(self.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i]*infoNCELoss_list_weights[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]*behavior_loss_list_weights[i]).sum()  
                

            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = (bprloss + args.reg * regLoss + args.beta*infoNCELoss) / args.batch

            epoch_loss = epoch_loss + loss.item()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()


#---round three---------------------------------------------------------------------------------------------
            cnt+=1

        return epoch_loss, user_embed, item_embed, user_embeds, item_embeds

    def testEpoch(self, data_loader, save=False):
        # 设置lagcl模型为评估模式，禁用dropout等训练特性
        self.lagcl_model.eval()
        
        epochHR, epochNDCG = [0]*2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds = self.model()

            # LAGCL模型测试获取用户和物品嵌入
            (user_embs, item_embs), _, _ = self.lagcl_model.process_one_subgraph()  
        cnt = 0
        tot = 0

        # 设置lagcl有关指标
        topk = args.metric_topk  
        rec_list = {} 

        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)  
            userEmbed = user_embed[user_compute] 
            itemEmbed = item_embed[item_compute]
           
            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)  

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)  
            epochHR = epochHR + hit  
            epochNDCG = epochNDCG + ndcg  #
            cnt += 1 
            tot += user.shape[0]

            # --------lagcl--------
            test_user = [f'userid_{id}' for id in user.tolist()]
            for u in test_user:
                user_embedding = user_embs[self.node_raw_idx_mapping[u]] 
                item_embeddings = item_embs 
               
                candidates = t.matmul(
                    user_embedding, 
                    item_embeddings.t()  
                ).detach().cpu().numpy() 

              
                for item in self.training_set_u[u].keys():
                    candidates[self.node_raw_idx_mapping[item] - self.lagcl_model.user_num] = -10e8  

               
                from metrics import find_k_largest, ranking_evaluation
                ids, scores = find_k_largest(topk, candidates) 
                item_names = [self.id2item[iid] for iid in ids]
                rec_list[u] = list(zip(item_names, scores)) 


    
        metrics_res = ranking_evaluation(self.test_set, rec_list, topk)
        
        lagcl_recall_20 = metrics_res['Recall']  
        lagcl_ndcg_20 = metrics_res['NDCG']  

        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}, lagcl_recall_20:{lagcl_recall_20}, lagcl_ndcg_20:{lagcl_ndcg_20}")


        return result_HR, result_NDCG, lagcl_recall_20, lagcl_ndcg_20

    def calcRes(self, pred_i, user_item1, user_item100): 
     
        hit = 0
        ndcg = 0

    
        for j in range(pred_i.shape[0]):

            _, shoot_index = t.topk(pred_i[j], args.shoot) 
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            if type(shoot)!=int and (user_item1[j] in shoot):  
                hit += 1  
                ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))  
            elif type(shoot)==int and (user_item1[j] == shoot):
                hit += 1  
                ndcg += np.reciprocal( np.log2( 0+2))
    
        return hit, ndcg  #int, float

    def sampleTestBatch(self, batch_user_id, batch_item_id):
       
        batch = len(batch_user_id)
        tmplen = (batch*100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()  
        user_item1 = batch_item_id 
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i] 
            negset = np.reshape(np.argwhere(sub_trainMat[i]==0), [-1])  
            pvec = self.labelP[negset] 
            pvec = pvec / np.sum(pvec)  
            
            random_neg_sam = np.random.permutation(negset)[:99]  
            user_item100_one_user = np.concatenate(( random_neg_sam, np.array([pos_item]))) 
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100

    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    def getModelName(self):  
        title = args.title
        ModelName = \
        args.point + \
        "_" + title + \
        "_" +  args.dataset +\
        "_" + modelTime + \
        "_lr_" + str(args.lr) + \
        "_reg_" + str(args.reg) + \
        "_batch_size_" + str(args.batch) + \
        "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    def saveHistory(self):  
        history = dict()
        history['loss'] = self.train_loss  
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        # 检查并创建目录
        history_dir = os.path.join('./History', args.dataset)
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        with open(os.path.join(history_dir, ModelName + '.his'), 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):  
        ModelName = self.modelName

        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            # 'lr': self.lr,
            'model': self.model,
            # 'reg': self.reg,
            'history': history,
            'user_embed': self.user_embed,
            'user_embeds': self.user_embeds,
            'item_embed': self.item_embed,
            'item_embeds': self.item_embeds,
        }

        # 检查并创建目录
        model_dir = os.path.join('./Model', args.dataset)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        t.save(params, savePath)

    def loadModel(self, loadPath):      
        ModelName = self.modelName
        # loadPath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        loadPath = loadPath
        checkpoint = t.load(loadPath)
        self.model = checkpoint['model']

        self.curEpoch = checkpoint['epoch'] + 1
        # self.lr = checkpoint['lr']
        # self.args.reg = checkpoint['reg']
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        # log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))

if __name__ == '__main__':
    print(args)
    my_model = Model()
    my_model.run()
    # my_model.test()

