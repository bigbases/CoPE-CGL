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
from data_stream_mixed import Streaming
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, train_node_classifier_batch, eval_node_classifier
from backbones.gcn import GCN



def main():
    parser = argparse.ArgumentParser()
    # Arguments for data.
    parser.add_argument('--dataset-name', type=str, default="corafull")
    parser.add_argument('--cls-per-task', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--result-path', type=str, default="./results")

    # Argumnets for CGL methods.
    parser.add_argument('--tim', action='store_true')
    parser.add_argument('--cgl-method', type=str, default="cgm")
    parser.add_argument('--cls-epoch', type=int, default=200) # Reduced for TPP prompt tuning speed
    parser.add_argument('--hidden', type=int, default=256) # TPP needs hidden dim info

    # Method specific args
    parser.add_argument('--lwf-args', type=str, default="{'lambda_dist': 10., 'T': 20.}")
    
    # TPP Specific Arguments
    # pe: probability of edge drop, pf: feature drop, prompts: num prompts
    parser.add_argument('--tpp-args', type=str, default="{'pe': 0.2, 'pf': 0.3, 'prompts': 1}")
    
    parser.add_argument('--taam-args', type=str, default="{'task_emb_dim': 64, 'num_heads': 3, 'rank': 4, 'h_dims': [256], 'sgc_k': 2, 'dropout': 0.0, 'batch_norm': False, 'linear_bias': True, 'lr': 0.005, 'weight_decay': 5e-4, 'prototype_hops': [0, 2], 'prototype_alpha': 0.1, 'prototype_steps': 10}")

    parser.add_argument('--IL', type=str, default="classIL")
    parser.add_argument('--batch', action='store_true')
    
    # =========================================================
    # [추가됨] SGC + Community CGL Specific Arguments
    # =========================================================
    parser.add_argument('--sgc-hops', type=int, default=2, help="Number of hops for SGC feature smoothing")
    parser.add_argument('--max-prototypes', type=int, default=10, help="Number of Prototypes")
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha")
    parser.add_argument('--adapter-lr', type=float, default=0.01, help="Learning rate for adapter and head")
    parser.add_argument('--adapter-wd', type=float, default=5e-4, help="Weight decay for adapter and head")

    # Others
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('--ind_task_ratio', type=float, default=0.9) # inductive ratio
    
    args = parser.parse_args()
    print(args)

    # Get file names. 
    result_file_name = get_result_file_name(args)
    task_file = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}.streaming")

    dataset = get_dataset(args)
    if os.path.exists(task_file):
        data_stream = torch.load(task_file)
    else:
        data_stream = Streaming(args.cls_per_task, dataset)
        torch.save(data_stream, task_file)

    
    # Data Stream Loading

    # data_stream = Streaming(args.cls_per_task, dataset, args.ind_task_ratio)
    data_stream = Streaming(args.cls_per_task, dataset, args.ind_task_ratio)
    #torch.save(data_stream, task_file)

    # Main Execution Loop
    if args.cgl_method in ["lwf", "cope_gl",'tpp','taam','TAAM']: # Add TPP to this list
        APs = []
        AFs = []
        mAPs = []
        tmm = []
        
        for i in range(args.repeat):
            #seed_everything(args.seed + i)
            ttt0 = time.time()
            
            # Initialize Backbone (shared)
            model = get_backbone_model(dataset, data_stream, args)
            
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
