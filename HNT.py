import os  
import torch  
import torch.nn as nn 
import torch.nn.functional as F  
import numpy as np  
import wandb 
from lagcl_encoder import LAGCLEncoder  
from Utils.loss_utils import bpr_loss, l2_reg_loss, InfoNCE  


class LAGCLLinkPredictionModel(torch.nn.Module):
    

    def __init__(
        self,
        graph,  
        user_num: int,  
        item_num: int,  
        hidden_dim: int, 
        n_hops: int, 
        noise_eps: float,  
        cl_tau: float, 
        cl_rate: float,  
        tail_k_threshold: int, 
        m_h_eps: float, 
        kl_eps: float,  
        l2_reg: float,  
        hesitation_threshold: float, 
        hesitation_weight: float, 
        agg_function: int,  
        neighbor_norm_type: str,  
        use_relation_rf: bool,  
        annealing_type: int, 
        lambda_gp: float, 
        agg_w_exp_scale: float,  
        device='cpu', 
    ):
      
        super().__init__()  
       
        self.noise_eps = noise_eps
        self.cl_tau = cl_tau
        self.cl_rate = cl_rate
        self.tail_k_threshold = tail_k_threshold
        self.m_h_eps = m_h_eps
        self.kl_eps = kl_eps
        self.l2_reg = l2_reg
        self.hesitation_threshold = hesitation_threshold
        self.hesitation_weight = hesitation_weight
        self.agg_function = agg_function
        self.neighbor_norm_type = neighbor_norm_type
        self.use_relation_rf = use_relation_rf
        self.annealing_type = annealing_type
        self.lambda_gp = lambda_gp
        self.agg_w_exp_scale = agg_w_exp_scale

       
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_dim = hidden_dim
        self.device = device

        
        self.graph = graph.to(device)
       
        self.node_ids = self.graph.ndata['node_feature']
       
        self.node_types = self.graph.ndata['node_type']
       
        self.is_user_node = self.node_types == 0
        
        self.global_head_mask = self.graph.ndata['node_degree'] > tail_k_threshold

        
        user_original_ids_in_graph = self.node_ids[self.is_user_node]
        item_original_ids_in_graph = self.node_ids[~self.is_user_node] - self.user_num

       
        user_is_head_in_graph = self.global_head_mask[self.is_user_node]
        item_is_head_in_graph = self.global_head_mask[~self.is_user_node]

        
        self.user_is_head_mask = torch.zeros(self.user_num, dtype=torch.bool, device=self.device)
        self.item_is_head_mask = torch.zeros(self.item_num, dtype=torch.bool, device=self.device)

        
        self.user_is_head_mask[user_original_ids_in_graph] = user_is_head_in_graph
        self.item_is_head_mask[item_original_ids_in_graph] = item_is_head_in_graph

        self.user_is_tail_mask = ~self.user_is_head_mask
        self.item_is_tail_mask = ~self.item_is_head_mask

        self._node_initial = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(user_num + item_num, hidden_dim, device=device)))

        self.negative_transfer_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(device)

        self._encoder = LAGCLEncoder(
            subgraph=graph,
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            noise_eps=noise_eps,
            cl_tau=cl_tau,
            cl_rate=cl_rate,
            tail_k_threshold=tail_k_threshold,
            m_h_eps=m_h_eps,
            kl_eps=kl_eps,
            agg_function=agg_function,
            neighbor_norm_type=neighbor_norm_type,
            use_relation_rf=use_relation_rf,
            agg_w_exp_scale=agg_w_exp_scale,
            device=device,
        )

    def split_user_item_embs(self, node_embs):
        user_node_indices_in_graph = torch.where(self.is_user_node)[0]
        item_node_indices_in_graph = torch.where(~self.is_user_node)[0]

        user_original_ids = self.node_ids[self.is_user_node]
        item_original_ids = self.node_ids[~self.is_user_node] - self.user_num

        user_embs = torch.zeros(self.user_num, self.hidden_dim, device=self.device)
        item_embs = torch.zeros(self.item_num, self.hidden_dim, device=self.device)

        user_embs[user_original_ids] = node_embs[user_node_indices_in_graph]
        item_embs[item_original_ids] = node_embs[item_node_indices_in_graph]

        return user_embs, item_embs

    def process_one_subgraph(self):
       
       
        x = self._node_initial[self.node_ids]

       
        node_embeddings, other_embs_dict = self._encoder(x)
       
        rec_user_emb, rec_item_emb = self.split_user_item_embs(node_embeddings)

        emb_h = other_embs_dict['head_true_drop_false']
        emb_nt = other_embs_dict['head_true_drop_true']
        emb_t = other_embs_dict['head_false_drop_true']

        
        rec_user_emb_h, rec_item_emb_h = self.split_user_item_embs(emb_h)
        rec_user_emb_t, rec_item_emb_t = self.split_user_item_embs(emb_t)
        rec_u_emb_nt, rec_i_emb_nt = self.split_user_item_embs(emb_nt)

        
        support_h = other_embs_dict['support_h']
        support_t = other_embs_dict['support_t'] 
        missing_info_h = other_embs_dict['missing_info_h']
        missing_info_t = other_embs_dict['missing_info_t']
       
        rec_user_missing_info_h, rec_item_missing_info_h = self.split_user_item_embs(missing_info_h)
        rec_user_missing_info_t, rec_item_missing_info_t = self.split_user_item_embs(missing_info_t)

        return (rec_user_emb, rec_item_emb), (support_h, support_t), {
           
            'user_head_true_drop_false': rec_user_emb_h,
            'item_head_true_drop_false': rec_item_emb_h,
            
            'user_head_true_drop_true': rec_u_emb_nt,
            'item_head_true_drop_true': rec_i_emb_nt,
           
            'user_head_false_drop_true': rec_user_emb_t,
            'item_head_false_drop_true': rec_item_emb_t,
           
            'user_missing_info_h': rec_user_missing_info_h,
            'item_missing_info_h': rec_item_missing_info_h,
            'user_missing_info_t': rec_user_missing_info_t,
            'item_missing_info_t': rec_item_missing_info_t,

        }

    def forward(self, data):
        
        user_idx, pos_idx, neg_idx = data
        pos_idx = torch.tensor(pos_idx, device=self.device) - self.user_num
        neg_idx = torch.tensor(neg_idx, device=self.device) - self.user_num
        # 用户索引 user_idx 本身就是从0开始的，无需转换
        user_idx = torch.tensor(user_idx, device=self.device)

       
        (rec_user_emb, rec_item_emb), (support_h, support_t), other_embs_dict = self.process_one_subgraph()

      
        u_emb_h = other_embs_dict['user_head_true_drop_false']
        u_emb_t = other_embs_dict['user_head_false_drop_true']
        u_emb_nt = other_embs_dict['user_head_true_drop_true']
      
        i_emb_h = other_embs_dict['item_head_true_drop_false']
        i_emb_t = other_embs_dict['item_head_false_drop_true']
        i_emb_nt = other_embs_dict['item_head_true_drop_true']
        
        u_missing_info_h = other_embs_dict['user_missing_info_h']
        u_missing_info_t = other_embs_dict['user_missing_info_t']
        i_missing_info_h = other_embs_dict['item_missing_info_h']
        i_missing_info_t = other_embs_dict['item_missing_info_t']

        
        u_knowledge_weights = torch.ones(self.user_num, 1, device=self.device)
        i_knowledge_weights = torch.ones(self.item_num, 1, device=self.device)

       
        if torch.any(self.user_is_tail_mask):
            u_knowledge_weights[self.user_is_tail_mask] = self.negative_transfer_net(u_missing_info_t[self.user_is_tail_mask])
        if torch.any(self.item_is_tail_mask):
            i_knowledge_weights[self.item_is_tail_mask] = self.negative_transfer_net(i_missing_info_t[self.item_is_tail_mask])

        
        denoised_rec_user_emb = rec_user_emb + (u_knowledge_weights - 1) * u_missing_info_t
        denoised_rec_item_emb = rec_item_emb + (i_knowledge_weights - 1) * i_missing_info_t
       

        
        batch_user_emb = denoised_rec_user_emb[user_idx]
        batch_pos_item_emb = denoised_rec_item_emb[pos_idx]
        
        batch_low_pos_item_emb = (u_knowledge_weights[user_idx]) * u_missing_info_t[user_idx]
        batch_neg_item_emb = denoised_rec_item_emb[neg_idx]

      
        task_loss = bpr_loss(user_emb=batch_user_emb,
                             pos_item_emb=batch_pos_item_emb,
                             low_pos_item_emb=batch_low_pos_item_emb,
                             neg_item_emb=batch_neg_item_emb)
       
        l2_loss = l2_reg_loss(self.l2_reg, batch_user_emb, batch_pos_item_emb)

       
        hesitation_loss = torch.tensor(0.0, device=self.device)
        if self.hesitation_weight > 0 and self.training:
           
            is_tail_user_in_batch = self.user_is_tail_mask[user_idx]
            knowledge_weights_in_batch = u_knowledge_weights[user_idx]
            hesitation_user_mask = (knowledge_weights_in_batch.squeeze() > self.hesitation_threshold) & is_tail_user_in_batch

            if torch.any(hesitation_user_mask):
                hesitation_user_indices = user_idx[hesitation_user_mask]
                hesitation_user_embs = denoised_rec_user_emb[hesitation_user_indices]

                scores = torch.matmul(hesitation_user_embs, denoised_rec_item_emb.t())

               
                for i, user_id in enumerate(hesitation_user_indices):
                  
                    user_pos_items_in_batch = pos_idx[user_idx == user_id]
                    if len(user_pos_items_in_batch) > 0:
                        scores[i, user_pos_items_in_batch] = -torch.inf

               
                weak_pos_item_indices = torch.argmax(scores, dim=1)
                weak_pos_item_embs = denoised_rec_item_emb[weak_pos_item_indices]

                
                hesitation_neg_item_embs = batch_neg_item_emb[hesitation_user_mask]

                hesitation_loss = bpr_loss(user_emb=hesitation_user_embs,
                                           pos_item_emb=weak_pos_item_embs,
                                           low_pos_item_emb=None,
                                           neg_item_emb=hesitation_neg_item_embs)
                hesitation_loss *= self.hesitation_weight
       

       
        L_kl_corr_user = L_kl_corr_item = torch.tensor(0.0, device=self.device) 
        if self._encoder.kl_eps > 0:
          
            L_kl_corr_user = torch.nn.functional.kl_div(
                input=u_emb_t[user_idx].log_softmax(dim=-1),
                target=u_emb_h[user_idx].softmax(dim=-1),
                reduction='batchmean',  
                log_target=False       
            ) * self._encoder.kl_eps  

            

        L_kl_corr = L_kl_corr_user + L_kl_corr_item  

        
        cl_loss = torch.tensor(0.0, device=self.device) 
        if self._encoder.cl_rate > 0: 
            cl_loss = self._encoder.cl_rate * self.cal_cl_loss(
                [user_idx, pos_idx], rec_user_emb, rec_item_emb)

       
        m_regularizer = torch.tensor(0.0, device=self.device) 
        if self._encoder.m_h_eps > 0: 
           
            support_vectors_to_regularize = support_t if self.use_relation_rf else support_h
           
            user_support_vectors = [layer_support[self.is_user_node] for layer_support in support_vectors_to_regularize]
            m_regularizer = self.cal_m_regularizer_loss(
                user_support_vectors,
                self.user_is_head_mask 
            ) * self._encoder.m_h_eps 

       
        loss = task_loss + l2_loss + L_kl_corr + cl_loss + m_regularizer + hesitation_loss

       
        return (rec_user_emb, rec_item_emb), loss

    def cal_cl_loss(self, idx, user_emb, item_emb):
       
        
        u_idx = torch.unique(idx[0]).to(self.device) 
        i_idx = torch.unique(idx[1]).to(self.device) 

        def exec_perturbed(x):
            random_noise = torch.rand_like(x).to(self.device)
            perturbed_x = x + torch.sign(x) * F.normalize(
                random_noise, dim=-1) * self._encoder.noise_eps
            return perturbed_x

        user_view_1 = exec_perturbed(user_emb)
        user_view_2 = exec_perturbed(user_emb)
        item_view_1 = exec_perturbed(item_emb)
        item_view_2 = exec_perturbed(item_emb)

        
        user_cl_loss = InfoNCE(user_view_1[u_idx],  user_view_2[u_idx],
                               self._encoder.cl_tau)   
        item_cl_loss = InfoNCE(item_view_1[i_idx],
                               item_view_2[i_idx], 
                               self._encoder.cl_tau) 

        
        return user_cl_loss + item_cl_loss

    def cal_m_regularizer_loss(self, support_out, head_mask):
       
        m_reg_loss = 0.0 
        for layer_support in support_out: 
           
            m_reg_loss += torch.mean(torch.norm(layer_support[head_mask], p=2, dim=1))

       
        return m_reg_loss

    def train_disc(self, embed_model, disc_model, optimizer_D,
                   disc_pseudo_real, optimizer_D_pseudo_real, cur_iter,
                   iter_num):
             
        disc_model.train()          
       
        embed_model.train()        
        for p in disc_model.parameters():
            p.requires_grad = True
        for p in disc_pseudo_real.parameters():
            p.requires_grad = True

        with torch.no_grad():
            _, _, other_embs_dict = embed_model.process_one_subgraph()
           
            u_emb_h = other_embs_dict['user_head_true_drop_false'] 
            u_emb_t = other_embs_dict['user_head_false_drop_true'] 
            u_emb_nt = other_embs_dict['user_head_true_drop_true'] 
            all_head_emb_h = u_emb_h[self.user_is_head_mask]
           
            all_emb_t_fake = u_emb_t 
            all_emb_nt = u_emb_nt 
            cell_mask = self.user_is_head_mask 
            
            noise_eps = 0.0
            if True:
                if self.annealing_type == 0:
                    noise_eps = self.noise_eps
                elif self.annealing_type == 1: 
                    annealing_stop_rate = 0.7 
                    rate = 1.0 - min(cur_iter / (annealing_stop_rate * iter_num + 1e-8), 1.0)
                    noise_eps = self.noise_eps * rate
                elif self.annealing_type == 2: 
                    annealing_stop_rate = 0.6 
                    annealing_rate = (0.01**(1.0 / (iter_num * annealing_stop_rate + 1e-8)))
                    noise_eps = self.noise_eps * (annealing_rate**cur_iter)

            def exec_perturbed(x, noise_eps):
                if noise_eps > 0: 
                    random_noise = torch.rand_like(x).to(self.device)
                    x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * noise_eps
                return x

            
            all_head_emb_h = exec_perturbed(all_head_emb_h, noise_eps=noise_eps) 
            all_emb_t_fake = exec_perturbed(all_emb_t_fake, noise_eps=noise_eps) 
            all_emb_nt = exec_perturbed(all_emb_nt, noise_eps=noise_eps)      

        
        prob_h = disc_model(all_head_emb_h) 
        prob_t = disc_model(all_emb_t_fake) 

       
        errorD = -prob_h.mean() 
        errorG = prob_t.mean()  

       
        def get_select_idx(max_value, select_num, strategy='uniform'):
            if max_value == 0: return torch.tensor([], dtype=torch.long) 
            select_idx = None
            if strategy == 'uniform': 
                
                select_idx = torch.randperm(max_value).repeat(
                    int(np.ceil(select_num / max_value)))[:select_num]
            elif strategy == 'random': 
                select_idx = np.random.randint(0, max_value, select_num)
            return select_idx

       
        def calc_gradient_penalty(netD, real_data, fake_data, lambda_gp):
            if not real_data.numel() or not fake_data.numel(): 
                 return torch.tensor(0.0, device=self.device)
            batch_size = real_data.size(0)
            alpha = torch.rand(batch_size, 1).to(self.device)
            alpha = alpha.expand(real_data.size()) 

            
            interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).detach().requires_grad_(True)
           
            disc_interpolates = netD(interpolates)

            import torch.autograd as autograd
           
            gradients = autograd.grad(outputs=disc_interpolates,
                                      inputs=interpolates,
                                      grad_outputs=torch.ones(
                                          disc_interpolates.size(), device=self.device),
                                      create_graph=True, 
                                      retain_graph=True, 
                                      only_inputs=True)[0] 
            gradients = gradients.view(gradients.size(0), -1) 

            gradient_penalty = (
                (gradients.norm(2, dim=1) - 1)**2).mean() * lambda_gp 
            return gradient_penalty

      
        gp_fake_data = all_emb_t_fake
       
        gp_real_data_indices = get_select_idx(len(all_head_emb_h), len(gp_fake_data), strategy='random')
        gp_real_data = all_head_emb_h[gp_real_data_indices] 
        gradient_penalty = calc_gradient_penalty(netD=disc_model,
                                                real_data=gp_real_data,
                                                fake_data=gp_fake_data,
                                                lambda_gp=self.lambda_gp)

       
        L_d = errorD + errorG + gradient_penalty
        
        optimizer_D.zero_grad() 
        L_d.backward()         
        optimizer_D.step()      

        pseudo_embs = all_emb_nt[cell_mask]      
        real_tail_embs = all_emb_nt[~cell_mask] 
 
        if len(pseudo_embs) > len(real_tail_embs):
            gp_fake_data_d2 = pseudo_embs
            gp_real_data_indices_d2 = get_select_idx(len(real_tail_embs), len(gp_fake_data_d2), strategy='random')
            gp_real_data_d2 = real_tail_embs[gp_real_data_indices_d2]
        else:
            gp_real_data_d2 = real_tail_embs
            gp_fake_data_indices_d2 = get_select_idx(len(pseudo_embs), len(gp_real_data_d2), strategy='random')
            gp_fake_data_d2 = pseudo_embs[gp_fake_data_indices_d2]

       
        L_gp2 = calc_gradient_penalty(netD=disc_pseudo_real,
                                      real_data=gp_real_data_d2,
                                      fake_data=gp_fake_data_d2,
                                      lambda_gp=self.lambda_gp)

       
        prob_t_with_miss = disc_pseudo_real(all_emb_nt) 
       
        errorR_pseudo = prob_t_with_miss[cell_mask].mean()
        
        errorR_real_tail = -prob_t_with_miss[~cell_mask].mean()

       
        L_d2 = errorR_pseudo + errorR_real_tail + L_gp2
      
        optimizer_D_pseudo_real.zero_grad()
        L_d2.backward()
        optimizer_D_pseudo_real.step()

       
        log = {
           
            'loss/disc1_errorD': errorD.item(),
            'loss/disc1_errorG': errorG.item(), 
           
            'loss/disc1_gp': gradient_penalty.item(),
            'loss/disc1_full': L_d.item(),
           
            'loss/disc2_full': L_d2.item(), 
            'loss/disc2_gp': L_gp2.item(), 
            'loss/disc2_errorR_pseudo': errorR_pseudo.item(), 
            'loss/disc2_errorR_real_tail': errorR_real_tail.item(), 
            'noise_eps': noise_eps, 
        }
       
        if os.environ.get('use_wandb'):
            wandb.log(log)

       
        return L_d

    def train_gen(self, embed_model, optimizer, disc_model, disc_pseudo_real):
        
        embed_model.train()      
        disc_model.train()
        disc_pseudo_real.train()

      
        for p in disc_model.parameters():
            p.requires_grad = False
        for p in disc_pseudo_real.parameters():
            p.requires_grad = False

        
        _, _, other_embs_dict = embed_model.process_one_subgraph()
        
        u_emb_h = other_embs_dict['user_head_true_drop_false']
        u_emb_t = other_embs_dict['user_head_false_drop_true'] 
        u_emb_nt = other_embs_dict['user_head_true_drop_true'] 
        
        all_emb_t_fake = u_emb_t[~self.user_is_head_mask]
        
        prob_t = disc_model(all_emb_t_fake) 
        
        L_disc1 = -prob_t.mean() * 0.1

        all_emb_nt_pseudo = u_emb_nt[self.user_is_head_mask]
        
        prob_t_with_miss = disc_pseudo_real(all_emb_nt_pseudo) 
       
        L_disc2 = -prob_t_with_miss.mean() * 0.1

       
        L_g = L_disc1 + L_disc2

       
        optimizer.zero_grad() 
        L_g.backward()       
        optimizer.step()     

      
        log = {
            'loss/gen_disc1': L_disc1.item(),
            'loss/gen_disc2': L_disc2.item(), 
            'loss/gen_full': L_g.item(),     
        }
        if os.environ.get('use_wandb'):
            wandb.log(log)

       
        return L_g
