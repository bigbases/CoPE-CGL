# CoPE-GL

# Abstract
> Continual graph learning (CGL) enables models to learn from a stream of graph-structured tasks while mitigating catastrophic forgetting. Existing methods, however, are typically tailored to a single evaluation regime—either a transductive setting, where each task introduces disjoint node classes within a shared graph, or an inductive setting, where tasks comprise separate graphs with overlapping classes—and they commonly depend on costly replay buffers or explicit task identifiers at inference time. We propose CoPE-GL (Community-aware Prototype Extraction for Graph Learning), a replay-free CGL framework built on a single principle: a parameter-free structural prior, obtained by precomputing multi-hop Laplacian smoothing over each task graph, can serve as a persistent, non-drifting anchor that simultaneously supports learning, memory, and inference. CoPE-GL instantiates this principle through three coupled components: (i) Topology-Infused Bottleneck Adaptation (TIBA), which trains an ultra-lightweight bottleneck adapter on top of the frozen structural prior, achieving millisecond-level task specialization without recursive message passing; (ii) community-aware prototype extraction, which partitions each task graph into communities and distills them into a small, fixed set of deterministic prototypes that capture fine-grained topological heterogeneity at zero training cost, unlike learnable single-centroid approaches ; and (iii) graph-level dynamic routing, which removes the need for task identifiers at inference time by matching each test node against the prototype memory and selecting the corresponding adapter and classifier via majority voting. Unlike prior work that reports results under only one regime, we provide a unified evaluation across transductive and inductive settings, spanning four transductive and three inductive benchmarks against eight state-of-the-art baselines, as well as mixed task streams that interleave the two regimes. CoPE-GL consistently matches or surpasses baselines in every regime while delivering significantly faster training and lower memory consumption.

# Dependencies
- Python 3.9
- torch_Geometric 2.1
- networkx 3.1
- igraph 0.9.9

# Dataset
Datasets can be obtained from [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) library. 

# Acknowledgment
This code is implemented based on CaT-CGL(https://github.com/superallen13/CaT-CGL.git) and CGLB(https://github.com/QueuQ/CGLB.git). Please refer to both for more baselines and implementation details.

# Run
Run ```sh run.sh```
