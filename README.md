# CoPE-CGL: Community-aware Prototype Extraction for Replay-Free Continual Graph Learning

# Abstract
> Continual graph learning (CGL) learns from a stream of graph-structured tasks while mitigating catastrophic forgetting. Existing methods are typically tailored to a single evaluation setting, either transductive (disjoint node classes within a shared graph) or inductive (task subgraphs that share nodes, edges, and classes with earlier tasks), and commonly depend on costly replay buffers or explicit task identifiers at inference time. We propose Community-aware Prototype Extraction for Continual Graph Learning (simply, CoPE-CGL), a replay-free CGL framework in which a parameter-free structural prior, precomputed by multi-hop Laplacian smoothing, anchors learning, memory, and inference. CoPE-CGL has three components: (i) topology-infused bottleneck adaptation (TIBA), a lightweight adapter trained on the frozen prior; (ii) community-aware prototype extraction, which distills each task into a few non-learned community-level prototypes; and (iii) graph-level dynamic routing, which infers the task of a test graph by a majority vote of its nodes against the prototype memory. As with prior task-free methods, the default protocol assumes that each test graph comes from a single learned task; a node-level variant relaxes this assumption. On four transductive and three inductive benchmarks and on mixed streams that interleave the two, CoPE-CGL outperforms the second-best of eleven representative baselines by up to 1.41, 6.32, and 9.33 percentage points of average performance (AP), runs 1.3–7.9× faster end-to-end than the most accurate competitor on each benchmark, and scales to the largest benchmark graph. At equal storage, it outperforms the strongest replay baseline by 15.82–33.48 points of AP while running 5.9–14.0× faster with zero forgetting. When a test graph mixes two tasks, and every method must route each node separately, its node-level variant gives the best AP among CGL methods. The design targets homophilic graphs with community structure. 

# Dependencies
- Python 3.9
- torch 2.4.1
- torch_Geometric 2.6
- networkx 3.1
- igraph 0.9.9

# Dataset
The *.streaming files are provided as compressed archives. Please extract them into the `data/` directory before running the project.

If the files are unavailable, download the dataset from this [Dropbox link](https://www.dropbox.com/scl/fi/xwmu1z8bkxpvwp8mo7f98/streaming.zip?rlkey=osuzzil4myf7hwk89jqll7wgf&st=0757ra29&dl=0).

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
