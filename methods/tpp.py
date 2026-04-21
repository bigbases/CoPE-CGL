import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree, add_self_loops
from torch_sparse import SparseTensor
import numpy as np

class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb

class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p

class ModelGrace(nn.Module):
    def __init__(self, model, num_hidden, num_proj_hidden, tau=0.5):
        super(ModelGrace, self).__init__()
        self.model = model
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def forward(self, x, edge_index):
        model = self._unwrap_model()

        num_nodes = x.size(0)
        adj_t = SparseTensor.from_edge_index(
            edge_index,
            sparse_sizes=(num_nodes, num_nodes)
        ).t().to(x.device)

        curr_x = x
        for layer in model.layers[:-1]:
            curr_x = layer(curr_x, adj_t)
            curr_x = F.relu(curr_x)

        output = curr_x
        z = F.elu(self.fc1(output))
        z = self.fc2(z)
        return z

    def sim(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss_batched(self, z1, z2, batch_size=4096):
        device = z1.device
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        num_nodes = z1.size(0)
        losses = []

        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)

            z1_batch = z1[start:end]  # [B, D]

            refl_sim = torch.exp(torch.mm(z1_batch, z1.t()) / self.tau)      # [B, N]
            between_sim = torch.exp(torch.mm(z1_batch, z2.t()) / self.tau)   # [B, N]

            numer = between_sim[:, start:end].diag()
            denom = refl_sim.sum(dim=1) + between_sim.sum(dim=1) - refl_sim[:, start:end].diag()

            loss_batch = -torch.log(numer / (denom + 1e-12))
            losses.append(loss_batch)

        return torch.cat(losses, dim=0)

    def loss(self, h1, h2, batch_size=4096):
        l1 = self.semi_loss_batched(h1, h2, batch_size=batch_size)
        l2 = self.semi_loss_batched(h2, h1, batch_size=batch_size)
        return (l1 + l2) * 0.5

def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def mask_edge(edge_index, drop_prob):
    E = edge_index.size(1)
    mask = torch.rand(E) >= drop_prob
    return edge_index[:, mask]

def traingrace_pyg(
    modelgrace,
    data,
    epochs=100,
    lr=1e-3,
    drop_edge_prob=0.2,
    drop_feature_prob=0.3,
    grace_batch_size=2048,
    max_grace_nodes=20000,
):
    modelgrace.train()
    optimizer = torch.optim.Adam(modelgrace.parameters(), lr=lr, weight_decay=1e-5)

    x, edge_index = data.x, data.edge_index

    if hasattr(data, 'train_mask') and data.train_mask is not None and data.train_mask.sum() > 0:
        candidate_idx = data.train_mask.nonzero(as_tuple=True)[0]
    else:
        candidate_idx = torch.arange(x.size(0), device=x.device)

    if candidate_idx.numel() == 0:
        return

    for _ in range(epochs):
        optimizer.zero_grad()

        edge_index_aug = mask_edge(edge_index, drop_edge_prob)
        x_aug = drop_feature(x, drop_feature_prob)

        z1 = modelgrace(x, edge_index)
        z2 = modelgrace(x_aug, edge_index_aug)

        if candidate_idx.numel() > max_grace_nodes:
            perm = torch.randperm(candidate_idx.numel(), device=candidate_idx.device)[:max_grace_nodes]
            anchor_idx = candidate_idx[perm]
        else:
            anchor_idx = candidate_idx

        z1_anchor = z1[anchor_idx]
        z2_anchor = z2[anchor_idx]

        if z1_anchor.size(0) > 0:
            loss = modelgrace.loss(
                z1_anchor,
                z2_anchor,
                batch_size=grace_batch_size
            ).mean()
            loss.backward()
            optimizer.step()

class TPP():
    def __init__(self, model, tasks, device, args):
        self.backbone = model
        self.tasks = tasks
        self.device = device
        self.args = args
        
        tpp_args = eval(args.get('tpp_args', "{}"))
        self.drop_edge_p = tpp_args.get('pe', 0.2)
        self.drop_feat_p = tpp_args.get('pf', 0.3)
        self.num_prompt = int(tpp_args.get('prompts', 1))
        self.d_data = tasks[0].x.shape[1]
        self.hidden = args['hidden'] if 'hidden' in args else 256
        
        self.prompts = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.prototypes = {} 
        self.ce = nn.CrossEntropyLoss()

    def get_embedding(self, x, edge_index):
        num_nodes = x.size(0)
        adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).t()
        adj_t = adj_t.to(self.device)
        
        curr_x = x
        for layer in self.backbone.layers[:-1]:
            curr_x = layer(curr_x, adj_t)
            curr_x = F.relu(curr_x)
            
        return curr_x

    def get_prototype(self, x, edge_index, train_mask, k=3):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = x
        for _ in range(k):
            out = torch.sparse.mm(
                torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0))), 
                out
            )
        if train_mask is not None and train_mask.sum() > 0:
            proto = torch.mean(out[train_mask], dim=0)
        else:
            proto = torch.mean(out, dim=0)
        return proto.detach()

    def get_task_id(self, x, edge_index, test_mask, k=3):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = x
        for _ in range(k):
             out = torch.sparse.mm(
                torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0))), 
                out
            )
        if test_mask is not None and test_mask.sum() > 0:
            test_proto = torch.mean(out[test_mask], dim=0).cpu()
        else:
            test_proto = torch.mean(out, dim=0).cpu()
            
        dists = []
        task_indices = sorted(self.prototypes.keys())
        for t_idx in task_indices:
            proto = self.prototypes[t_idx].cpu()
            dist = torch.norm(proto - test_proto)
            dists.append(dist)
        min_idx = torch.argmin(torch.tensor(dists))
        return task_indices[min_idx]

    def observer(self, epochs, IL):
        tasks = self.tasks
        performance_matrix = torch.zeros(len(tasks), len(tasks))
        APs = []
        
        for k in range(len(tasks)):
            task = tasks[k].to(self.device)
            x, edge_index, y = task.x, task.edge_index, task.y
            train_mask = task.train_mask
            
            curr_classes = torch.unique(y)
            n_cls = len(curr_classes)
            
            if n_cls == 0:
                n_cls = self.args.get('cls_per_task', 2)
                
            y_mapped = y.clone()
            for i, c in enumerate(curr_classes):
                y_mapped[y == c] = i
                
            cls_head = nn.Linear(self.hidden, n_cls).to(self.device)
            nn.init.xavier_uniform_(cls_head.weight)
            self.heads.append(cls_head)
            
            if k == 0:
                print("Pretraining Backbone with GRACE...")
                grace = ModelGrace(self.backbone, self.hidden, self.hidden * 2).to(self.device)
                traingrace_pyg(grace, task, epochs=epochs, drop_edge_prob=self.drop_edge_p, drop_feature_prob=self.drop_feat_p)
                self.backbone.eval()
                
                optimizer = torch.optim.Adam(cls_head.parameters(), lr=0.01, weight_decay=5e-4)
                
            else:
                if self.num_prompt < 2:
                    prompt = SimplePrompt(self.d_data).to(self.device)
                else:
                    prompt = GPFplusAtt(self.d_data, self.num_prompt).to(self.device)
                self.prompts.append(prompt)
                
                optimizer = torch.optim.Adam([
                    {'params': prompt.parameters()},
                    {'params': cls_head.parameters()}
                ], lr=0.01, weight_decay=5e-4)

            for epoch in range(epochs):
                cls_head.train()
                if k > 0: self.prompts[k-1].train()
                
                optimizer.zero_grad()
                
                if k == 0:
                    feat = x
                else:
                    feat = self.prompts[k-1].add(x)
                
                with torch.no_grad():
                    emb = self.get_embedding(feat, edge_index)
                
                logits = cls_head(emb)
                
                if train_mask is not None and train_mask.sum() > 0:
                    loss = self.ce(logits[train_mask], y_mapped[train_mask])
                    loss.backward()
                    optimizer.step()
            
            with torch.no_grad():
                self.prototypes[k] = self.get_prototype(x, edge_index, train_mask)

            print(f"Eval Task {k}:", end=" ")
            accs = []
            for t_id in range(k + 1):
                test_task = tasks[t_id].to(self.device)
                if hasattr(test_task, 'test_mask') and test_task.test_mask is not None:
                    test_mask = test_task.test_mask
                elif hasattr(test_task, 'val_mask') and test_task.val_mask is not None:
                    test_mask = test_task.val_mask
                else:
                    test_mask = torch.ones(test_task.num_nodes, dtype=torch.bool, device=self.device)

                if k == 0:
                    pred_task_id = 0
                else:
                    pred_task_id = self.get_task_id(test_task.x, test_task.edge_index, test_mask)
                
                if pred_task_id > len(self.heads) - 1: pred_task_id = len(self.heads) - 1
                
                curr_head = self.heads[pred_task_id]
                curr_prompt = self.prompts[pred_task_id-1] if pred_task_id > 0 else None
                
                curr_head.eval()
                if curr_prompt: curr_prompt.eval()
                
                with torch.no_grad():
                    tx = test_task.x
                    if curr_prompt:
                        tx = curr_prompt.add(tx)
                    
                    emb = self.get_embedding(tx, test_task.edge_index)
                    
                    logits = curr_head(emb)
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
