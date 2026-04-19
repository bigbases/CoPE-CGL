# DL and GL
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor

# Own modules
from methods.replay import Replay
from backbones.gcn import GCN
from backbones.encoder import Encoder
from methods.utility import get_graph_class_ratio
from backbones.gnn import train_node_classifier

# Utilities
from torch_geometric.loader import NeighborLoader
from .utility import *
import random

class CGM(Replay):
    def __init__(self, model, tasks, budget, m_update, device, args):
        super().__init__(model, tasks, budget, m_update, device)
        self.n_encoders = args['n_encoders']
        self.feat_lr = args['feat_lr']
        self.hid_dim = args['hid_dim']
        self.emb_dim = args['emb_dim']
        self.n_layers = args['n_layers']
        self.feat_init = args.get("feat_init", "randomChoice")
        self.hop = args['hop']
        self.activation = args['activation']
        
    def memorize(self, task, budgets):
        if not isinstance(budgets, list):
            budgets = [budgets] * len(task.classes)

        labels_cond = []
        for i, cls in enumerate(task.classes):
            labels_cond += [cls] * budgets[i]
        labels_cond = torch.tensor(labels_cond)

        # 🚨 [핵심 수정 부분] 무조건 sum(budgets)를 쓰지 않고, 
        # 실제로 현재 태스크를 위해 생성할 노드의 총 개수(len(labels_cond))를 사용합니다!
        actual_budget = len(labels_cond)
        
        # 300이 아닌 실제 필요한 크기(예: 17)만큼 텐서 생성
        feat_cond = torch.nn.Parameter(torch.FloatTensor(actual_budget, task.num_features))
        feat_cond = self._initialize_feature(task, budgets, feat_cond, self.feat_init)

        replayed_graph = self._condense(task, feat_cond, labels_cond, actual_budget)
        return replayed_graph
    
    def _initialize_feature(self, task, budgets, feat_cond, method="randomChoice"):
        if method == "randomNoise":
            torch.nn.init.xavier_uniform_(feat_cond)
            
        elif method == "randomChoice":
            sampled_ids = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                
                target_budget = budgets[i]
                
                if len(ids_at_cls) == 0:
                    all_ids = task.train_mask.nonzero(as_tuple=True)[0].tolist()
                    sampled_ids += random.choices(all_ids, k=target_budget)
                elif len(ids_at_cls) < target_budget:
                    sampled_ids += random.choices(ids_at_cls, k=target_budget)
                else:
                    sampled_ids += random.sample(ids_at_cls, k=target_budget)
                    
            sampled_feat = task.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat) # 이제 크기가 완벽하게 일치합니다.
            
        elif method == "kMeans":
            sampled_ids = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                
                target_budget = budgets[i]
                
                if len(ids_at_cls) == 0:
                    all_ids = task.train_mask.nonzero(as_tuple=True)[0].tolist()
                    sampled_ids += random.choices(all_ids, k=target_budget)
                elif len(ids_at_cls) < target_budget:
                    sampled_ids += random.choices(ids_at_cls, k=target_budget)
                else:
                    queried_ids = query(task, ids_at_cls, target_budget, self.device)
                    if len(queried_ids) < target_budget:
                        queried_ids += random.choices(ids_at_cls, k=(target_budget - len(queried_ids)))
                    sampled_ids += queried_ids
                    
            sampled_feat = task.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)
            
        return feat_cond

    # 🚨 파라미터로 sum(budgets) 리스트 대신 actual_budget 정수를 받도록 수정
    def _condense(self, task, feat_cond, labels_cond, actual_budget):
        # SparseTensor 생성 시에도 300 고정이 아닌 실제 개수 기반으로 생성
        self_loops = SparseTensor.eye(actual_budget, actual_budget).t()
        opt_feat = torch.optim.Adam([feat_cond], lr=self.feat_lr)

        cls_train_masks = []
        for cls in task.classes:
            cls_train_masks.append((task.y == cls).logical_and(task.train_mask))   
        
        encoder = Encoder(task.num_features, self.hid_dim, self.emb_dim, self.n_layers, self.hop, self.activation).to(self.device)
        for _ in range(self.n_encoders):
            encoder.initialize()
            with torch.no_grad():
                emb_real = encoder.encode(task.x.to(self.device), task.adj_t.to(self.device))
                emb_real = F.normalize(emb_real)
            emb_cond = encoder.encode_without_e(feat_cond.to(self.device))
            emb_cond = F.normalize(emb_cond)

            loss = torch.tensor(0.).to(self.device)
            for i, cls in enumerate(task.classes):
                real_emb_at_class = emb_real[cls_train_masks[i]]
                cond_emb_at_class = emb_cond[labels_cond == cls]
                
                if real_emb_at_class.size(0) == 0 or cond_emb_at_class.size(0) == 0:
                    continue
                    
                dist = torch.mean(real_emb_at_class, 0) - torch.mean(cond_emb_at_class, 0)
                loss += torch.sum(dist ** 2)
        
            # 🚨 [최종 수정 부분] 학습할 데이터가 존재하여 loss가 업데이트(requires_grad=True) 되었을 때만 역전파!
            if loss.requires_grad:
                opt_feat.zero_grad()
                loss.backward()
                opt_feat.step()

        replayed_graph = Data(x=feat_cond.detach().cpu(), 
                              y=labels_cond.cpu(), 
                              adj_t=self_loops)
        replayed_graph.train_mask = torch.ones(actual_budget, dtype=torch.bool)
        return replayed_graph