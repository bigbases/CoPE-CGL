# CoPE-GL

# Abstract
> Continual graph learning (CGL) learns from a stream of graph-structured tasks while mitigating catastrophic forgetting. Existing methods, however, are typically tailored to a single evaluation regime—either transductive (disjoint node classes within a shared graph) or inductive (separate graphs with overlapping classes)—and commonly depend on costly replay buffers or explicit task identifiers at inference time. We propose CoPE-GL (Community-aware Prototype Extraction for Graph Learning), a replay-free CGL framework built on a single principle: a parameter-free structural prior, obtained by precomputing multi-hop Laplacian smoothing, serves as a persistent anchor that simultaneously supports learning, memory, and inference. CoPE-GL realizes this principle through three coupled components: (i) topology-infused bottleneck adaptation (TIBA), a lightweight bottleneck adapter trained on top of the frozen prior; (ii) community-aware prototype extraction, which distills each task into a small set of deterministic community-level prototypes; and (iii) graph-level dynamic routing, which identifies the task at inference time by matching test nodes against the prototype memory via majority voting. On four transductive and three inductive benchmarks against eight state-of-the-art baselines, plus mixed task streams that interleave the two regimes, CoPE-GL outperforms the second-best baseline by up to 4.26%, 5.74%, and 9.33% AP under transductive, inductive, and mixed settings, respectively, with up to a 7.9× training speedup (5.23× in the mixed setting) and graceful scaling to the largest benchmark graph.

# Dependencies
- Python 3.9
- torch_Geometric 2.1
- networkx 3.1
- igraph 0.9.9

# Dataset
The *.streaming files are provided as compressed archives. Please extract them into the `data/` directory before running the project.

After extraction, data streams can be constructed using the `Streaming` class implemented in `data_stream.py` and `data_stream_mixed.py`.

Alternatively, the datasets can be obtained via the PyTorch Geometric library:
https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html


# Acknowledgment
This code is implemented based on CaT-CGL(https://github.com/superallen13/CaT-CGL.git) and CGLB(https://github.com/QueuQ/CGLB.git). Please refer to both for more baselines and implementation details.

# Run
- Transductive Setting
```sh run_transductive.sh```

- Inductive Setting
```sh run_inductive.sh```

- Mixed Setting
```sh run_mixed.sh```
