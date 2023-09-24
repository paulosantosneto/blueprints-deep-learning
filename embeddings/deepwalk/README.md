`Warning` This is my implementation of the article ['DeepWalk: Online Learning of Social Representations'](https://arxiv.org/pdf/1403.6652.pdf). Therefore, it is not an official repository of it. Any code improvement recommendations are welcome.

**To Train**
```
python3 deepwalk.py --mode train --epochs 50 --embedding_size 100 --number_nodes 50 --percent_edges 0.1
```

**To Infer**
```
python3 deepwalk.py --mode inference --model_load_path weights_50.pth --model_config_path parameters.json --node 0 --most_similar 5
```

**Other options**

- `--T` walk length of a random walk;
- `--gamma` number of random walks per node;
- `--w` controls the extent of the neighborhood or context in relation to the central node;
- `--epochs` defines the number of training epochs;
- `--embedding_size` defines the size of the latent/embedding space;
- `--mode` allows you to select between training or inference;
- `--number_nodes` number of nodes in graph;
- `--percent_edges` percentage of connections between nodes;
- `--plot_graph` boolean argument for plotting the graph;
- `--plot_pca` boolean argument to visualize the latent space generated using principal component analysis;
- `--plot_loss` boolean argument to visualize the performance of the cost function;
- `--save_model` enables the saving of parameters and weights for later inference;
- `--device` allows processing using GPU;
- `--most_similar` defines a range of the most similar nodes.

## Implementation aspects

- `Log Softmax` to transform logits into probabilities;
- `Cross-Entropy Loss` as a cost function.

## Simple Ilustration of Random Walk

![](https://github.com/paulosantosneto/GNNs/blob/main/embeddings/deepwalk/figures/random_walk.png)
