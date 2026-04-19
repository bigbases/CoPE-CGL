# For graph learning
import torch
from torch_geometric import seed_everything

# Utility
import os
import argparse
import sys
import numpy as np
from utilities import *
import time
from data_stream import Streaming
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, train_node_classifier_batch, eval_node_classifier
from torch_sparse import SparseTensor
from backbones.gcn import GCN
from backbones.sgc import SGC
import psutil


def main():
    parser = argparse.ArgumentParser()
    # Arguments for data.
    parser.add_argument('--dataset-name', type=str, default="corafull")
    parser.add_argument('--cls-per-task', type=int, default=3)
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--result-path', type=str, default="./results")

    # Argumnets for CGL methods.
    parser.add_argument('--tim', action='store_true')
    parser.add_argument('--cgl-method', type=str, default="cgm")
    parser.add_argument('--cls-epoch', type=int, default=200) 
    parser.add_argument('--hidden', type=int, default=256) 

    # Method specific args
    parser.add_argument('--lwf-args', type=str, default="{'lambda_dist': 10., 'T': 20.}")
    
    # TPP Specific Arguments
    parser.add_argument('--tpp-args', type=str, default="{'pe': 0.2, 'pf': 0.3, 'prompts': 1}")
    
    parser.add_argument('--taam-args', type=str, default="{'task_emb_dim': 64, 'num_heads': 3, 'rank': 4, 'h_dims': [256], 'sgc_k': 2, 'dropout': 0.0, 'batch_norm': False, 'linear_bias': True, 'lr': 0.005, 'weight_decay': 5e-4, 'prototype_hops': [0, 2], 'prototype_alpha': 0.1, 'prototype_steps': 10}")
    
    parser.add_argument('--sgc-hops', type=int, default=2, help="Number of hops for SGC feature smoothing")
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha")
    parser.add_argument('--max-prototypes', type=int, default=10, help="Number of Prototypes")
    parser.add_argument('--adapter-lr', type=float, default=0.01, help="Learning rate for adapter and head")
    parser.add_argument('--adapter-wd', type=float, default=5e-4, help="Weight decay for adapter and head")
    
    parser.add_argument('--use-topo', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-bottleneck', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-community', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-voting', action=argparse.BooleanOptionalAction)

    parser.add_argument('--IL', type=str, default="classIL")
    parser.add_argument('--batch', action='store_true')

    # Others
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--rewrite', action='store_true')
    
    args = parser.parse_args()
    print(args)

    # Get file names.
    result_file_name = get_result_file_name(args)
    dataset_li = torch.load(f"./data/streaming/{args.dataset_name}.streaming") #.tasks
    
    for kk, pyg_graph in enumerate(dataset_li):
        pyg_graph.adj_t = SparseTensor(
            row=pyg_graph.edge_index[0],
            col=pyg_graph.edge_index[1],
            sparse_sizes=(pyg_graph.num_nodes, pyg_graph.num_nodes)
        )
        pyg_graph.x = pyg_graph.x.float()
        pyg_graph.y = pyg_graph.y.long()
        pyg_graph.classes = pyg_graph.y[pyg_graph.train_mask].unique().tolist()
        dataset_li[kk] = pyg_graph

    args.cls_per_task = len(pyg_graph.y[pyg_graph.train_mask].unique().tolist())
    

    if args.cls_per_task == 0:
        args.cls_per_task = parser.get_default('cls_per_task')
    
    class Streaming():
        def __init__(self, cls_per_task, tasks):
            self.cls_per_task = cls_per_task
            self.tasks = tasks
            self.n_tasks = len(self.tasks)

    data_stream = Streaming(args.cls_per_task, dataset_li)
    
    all_classes = set()
    for g in dataset_li:
        all_classes.update(g.y.unique().tolist())
    num_total_classes = max(all_classes) + 1 


    # Main Execution Loop
    if args.cgl_method in ["lwf", "cope_gl"]: # Add TPP to this list
        APs = []
        AFs = []
        mAPs = []
        tmm = []
        
        for i in range(args.repeat):
            seed_everything(args.seed + i)
            ttt0 = time.time()
            
            model = SGC(dataset_li[0].x.shape[1], 256, num_total_classes, 2).to(args.device)
            
            # Initialize Method (TPP)
            cgl_model = get_cgl_model(model, data_stream, args)
            
            # Run Training & Eval
            AP, mAP, AF = cgl_model.observer(args.cls_epoch, args.IL)
            
            ttt1 = time.time() - ttt0
            tmm.append(ttt1)
            APs.append(AP)
            AFs.append(AF)
            mAPs.append(mAP)
        
        print(f"\nAP: {np.mean(APs):.2f}±{np.std(APs, ddof=1):.2f}", flush=True)
        print(f"mAP: {np.mean(mAPs):.2f}±{np.std(mAPs, ddof=1):.2f}", flush=True)
        print(f"BWT: {np.mean(AFs):.2f}±{np.std(AFs, ddof=1):.2f}", flush=True)
        print(f"Time: {np.mean(tmm):.2f}±{np.std(tmm, ddof=1):.2f}", flush=True)
        
    else:
        # Existing memory-bank based logic (ER, GEM etc.)
        pass 

if __name__ == '__main__':
    main()