import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, add_self_loops
import numpy as np
import igraph as ig

class FeatureAdapter(nn.Module):
    def __init__(self, dim, alpha):
        super().__init__()
        bottleneck = min(max(dim // 4, 16), 128)  #params
        self.alpha = alpha
        
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim)
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return x + self.alpha * self.net(x)


class SGCCommunityCGL():
    def __init__(self, tasks, device, sgc_hops, alpha, max_proto, args, cd_algo="louvain"):
        self.tasks = tasks
        self.device = device
        self.args = args 
        self.method = cd_algo
        self.alpha = alpha #1.0
        
        self.max_prototypes = 10 
        
        self.sgc_hops = sgc_hops 
        self.d_data = tasks[0].x.shape[1] 
        
        self.adapters = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.community_prototypes = {} 
        self.ce = nn.CrossEntropyLoss()

    def get_sgc_embedding(self, x, edge_index, k=2):
        num_nodes = x.size(0)
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        adj_t = torch.sparse_coo_tensor(edge_index_loop, norm, (num_nodes, num_nodes)).to(self.device)
        
        out = x
        # laplacian smoothing
        for _ in range(k):
            out = torch.sparse.mm(adj_t, out)
            
        return out

    @torch.no_grad()
    def extract_community_prototypes(self, task, train_mask, z_features):
        x, edge_index, y = task.x, task.edge_index, task.y
        num_nodes = x.size(0)
        
            
        edge_index_cpu = edge_index.detach().cpu()
        
        g = ig.Graph(n=int(num_nodes), edges=list(zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())), directed=False)
        g.simplify(multiple=True, loops=False)
        if self.method == "louvain":
            communities = g.community_multilevel()
        elif self.method == 'leiden':
            communities = g.community_leiden(objective_function="modularity")
        elif self.method == 'LP':
            communities = g.community_label_propagation()
        elif self.method == "infomap":
            # 주의: infomap은 igraph 파이썬 API에서 인자 이름이 'edge_weights' 입니다.
            communities = g.community_infomap() 
        elif self.method == "fastgreedy":
            communities = g.community_fastgreedy().as_clustering()
        else:
            raise NotImplementedError("Not supported algorithm.")
        
        if train_mask is not None and train_mask.sum() > 0:
            valid_indices = set(train_mask.nonzero(as_tuple=True)[0].tolist())
        else:
            valid_indices = set(range(num_nodes))

        comm_sizes = []
        valid_comms = []
        for comm in communities:
            comm_valid_nodes = [n for n in comm if n in valid_indices]
            if len(comm_valid_nodes) > 0:
                valid_comms.append(comm_valid_nodes)
                comm_sizes.append(len(comm_valid_nodes))

        prototypes = []
        if len(comm_sizes) > 0:
            max_prototypes = self.max_prototypes
            sorted_indices = torch.argsort(torch.tensor(comm_sizes), descending=True)
            keep_indices = sorted_indices[:max_prototypes]
            
            for idx in keep_indices:
                nodes_t = torch.tensor(valid_comms[idx.item()], dtype=torch.long, device=self.device)
                comm_proto = torch.mean(z_features[nodes_t], dim=0)
                prototypes.append(comm_proto)

        if len(prototypes) == 0:
            if train_mask is not None and train_mask.sum() > 0:
                prototypes.append(torch.mean(z_features[train_mask], dim=0))
            else:
                prototypes.append(torch.mean(z_features, dim=0))

        return torch.stack(prototypes).detach()

    @torch.no_grad()
    def infer_task_id(self, z_test, test_mask):
        if test_mask is not None and test_mask.sum() > 0:
            test_feats = z_test[test_mask]
        else:
            test_feats = z_test
            
        task_indices = sorted(self.community_prototypes.keys())
        N = test_feats.size(0)
        
        votes = torch.zeros(N, len(task_indices), device=self.device)
        test_feats_norm = F.normalize(test_feats, p=2, dim=1)
        
        for t_idx in task_indices:
            protos = self.community_prototypes[t_idx].to(self.device)
            protos_norm = F.normalize(protos, p=2, dim=1)
            
            sims = torch.mm(test_feats_norm, protos_norm.t())
            best_sims, _ = torch.max(sims, dim=1)
            votes[:, t_idx] = best_sims
            
        best_tasks = torch.argmax(votes, dim=1)
        vote_counts = torch.bincount(best_tasks, minlength=len(task_indices))
        predicted_task = torch.argmax(vote_counts).item()
        
        return predicted_task

    def observer(self, epochs, IL):
        tasks = self.tasks
        performance_matrix = torch.zeros(len(tasks), len(tasks))
        APs = []
        
        for k in range(len(tasks)):
            task = tasks[k].to(self.device)
            x, edge_index, y = task.x, task.edge_index, task.y
            train_mask = getattr(task, 'train_mask', None)
            
            curr_classes = torch.unique(y)
            n_cls = len(curr_classes)
            if n_cls == 0: n_cls = self.args.get('cls_per_task', 2)
                
            y_mapped = y.clone()
            for i, c in enumerate(curr_classes):
                y_mapped[y == c] = i
                
            adapter = FeatureAdapter(self.d_data, self.alpha).to(self.device)
            cls_head = nn.Linear(self.d_data, n_cls).to(self.device)
            nn.init.xavier_uniform_(cls_head.weight)
            
            self.adapters.append(adapter)
            self.heads.append(cls_head)
            
            optimizer = torch.optim.Adam([
                {'params': adapter.parameters()},
                {'params': cls_head.parameters()}
            ], lr=0.01, weight_decay=5e-4)

            with torch.no_grad():
                z_base = self.get_sgc_embedding(x, edge_index, k=self.sgc_hops)

            for epoch in range(epochs):
                adapter.train()
                cls_head.train()
                optimizer.zero_grad()
                
                z_adapted = adapter(z_base)
                logits = cls_head(z_adapted)
                
                if train_mask is not None and train_mask.sum() > 0:
                    loss = self.ce(logits[train_mask], y_mapped[train_mask])
                    loss.backward()
                    optimizer.step()
            
            self.community_prototypes[k] = self.extract_community_prototypes(task, train_mask, z_base)

            print(f"Eval Task {k}:", end=" ")
            accs = []
            for t_id in range(k + 1):
                test_task = tasks[t_id].to(self.device)
                test_mask = getattr(test_task, 'test_mask', torch.ones(test_task.num_nodes, dtype=torch.bool, device=self.device))
                
                with torch.no_grad():
                    z_test = self.get_sgc_embedding(test_task.x, test_task.edge_index, k=self.sgc_hops)
                    
                    if k == 0:
                        pred_task_id = 0
                    else:
                        pred_task_id = self.infer_task_id(z_test, test_mask)
                    
                    if pred_task_id > len(self.heads) - 1: pred_task_id = len(self.heads) - 1
                    
                    curr_adapter = self.adapters[pred_task_id]
                    curr_head = self.heads[pred_task_id]
                    
                    curr_adapter.eval()
                    curr_head.eval()
                    
                    z_adapted = curr_adapter(z_test)
                    logits = curr_head(z_adapted)
                    preds = logits.argmax(dim=1)
                
                t_y = test_task.y
                t_classes = torch.unique(t_y)
                t_y_mapped = t_y.clone()
                for i, c in enumerate(t_classes):
                    t_y_mapped[t_y == c] = i
                    
                mask_sum = int(test_mask.sum())
                if mask_sum > 0:
                    correct = preds[test_mask].eq(t_y_mapped[test_mask]).sum().item()
                    acc = correct / mask_sum * 100
                else:
                    acc = 0.0
                
                accs.append(acc)
                performance_matrix[k, t_id] = acc
                print(f"T{t_id} {acc:.2f}", end="|")
                test_task.to('cpu')

            AP = sum(accs) / len(accs) if len(accs) > 0 else 0
            print(f" AP: {AP:.2f}")
            APs.append(AP)
            
        AF = 0
        if len(tasks) > 1:
            for t in range(len(tasks)-1):
                AF += performance_matrix[len(tasks)-1, t] - performance_matrix[t, t]
            AF = AF / (len(tasks) - 1)

        return APs[-1], np.mean(APs), AF