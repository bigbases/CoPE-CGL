import torch
import random

from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor
from progressbar import ProgressBar

class Streaming():
    def __init__(self, cls_per_task, dataset, ind_task_ratio=0.9, internal_old_ratio=0.5):

        self.cls_per_task = cls_per_task
        self.ind_task_ratio = ind_task_ratio
        self.internal_old_ratio = internal_old_ratio
        self.tasks = self.prepare_tasks(dataset)
        self.n_tasks = len(self.tasks)
        
    def prepare_tasks(self, dataset):
        graph = dataset[0]
        tasks = []
        n_tasks = int(dataset.num_classes / self.cls_per_task)

        if not hasattr(graph, 'n_id'):
            graph.n_id = torch.arange(graph.num_nodes)


        num_ind_tasks = int(n_tasks * self.ind_task_ratio)
        num_trans_tasks = n_tasks - num_ind_tasks
        

        task_types = ['inductive'] * num_ind_tasks + ['transductive'] * num_trans_tasks
        random.shuffle(task_types)
        

        if task_types[0] == 'inductive':
            trans_idx = task_types.index('transductive')
            task_types[0], task_types[trans_idx] = task_types[trans_idx], task_types[0]
            
        print(f"Randomized Task Sequence: {task_types}")
        # =====================================================

        seen_nodes = torch.tensor([], dtype=torch.long)

        for k in ProgressBar()(range(n_tasks)): 
            start_cls = k * self.cls_per_task
            classes = list(range(start_cls, start_cls + self.cls_per_task))
            
            class_tensor = torch.tensor(classes, device=graph.y.device)
            current_new_nodes = torch.isin(graph.y, class_tensor).nonzero(as_tuple=False).squeeze()
            
            if current_new_nodes.dim() == 0:
                current_new_nodes = current_new_nodes.unsqueeze(0)
            
            current_task_type = task_types[k]
            

            if current_task_type == 'transductive':
                subset = current_new_nodes
                
            else:
                n_new_avail = len(current_new_nodes)
                n_old_avail = len(seen_nodes)
                target_r = self.internal_old_ratio
                
                limit_by_old = n_old_avail / target_r if target_r > 0 else 0
                limit_by_new = n_new_avail / (1 - target_r) if target_r < 1 else 0
                total_samples = int(min(limit_by_old, limit_by_new))
                
                n_old_select = int(total_samples * target_r)
                n_new_select = total_samples - n_old_select
                
                perm_old = torch.randperm(n_old_avail)[:n_old_select]
                perm_new = torch.randperm(n_new_avail)[:n_new_select]
                
                selected_old = seen_nodes[perm_old]
                selected_new = current_new_nodes[perm_new]
                
                subset = torch.cat([selected_old, selected_new])
            # =====================================================

            seen_nodes = torch.cat([seen_nodes, current_new_nodes]).unique()

            if len(subset) == 0:
                continue

            subgraph = graph.subgraph(subset)
            
            subgraph.new_nodes_mask = torch.isin(subgraph.n_id, current_new_nodes)

            transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
            subgraph = transform(subgraph)
            
            subgraph.train_mask = subgraph.train_mask & subgraph.new_nodes_mask
            subgraph.val_mask   = subgraph.val_mask   & subgraph.new_nodes_mask
            subgraph.test_mask  = subgraph.test_mask  & subgraph.new_nodes_mask
            

            edge_index = subgraph.edge_index
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes))
            subgraph.adj_t = adj.t().to_symmetric() 
            subgraph.task_id = k
            subgraph.classes = classes
            subgraph.task_type = current_task_type

            tasks.append(subgraph)
            
        return tasks
