`Warning` This is my implementation of the article ['node2vec: Scalable Feature Learning for Networks'](https://arxiv.org/pdf/1607.00653.pdf)[1]. Therefore, it is not an official repository of it. Any code improvement recommendations are welcome.

**To Train**
```
python3 node2vec.py --mode train --epochs 50 --embedding_size 100 --number_nodes 50 --percent_edges 0.1 --p 0.2 --q 0.8
```

**To Infer**
```
python3 node2vec.py --mode inference --model_load_path weights_50.pth --model_config_path parameters.json --node 0 --most_similar 5
```

**Arguments**

- `--l` walk length of a random walk;
- `--r` numver of random walks per node;
- `--k` controls the extent of the neighborhood or context in relation to the central node;
- `--p` out parameter;
- `--q` in-out parameter;
- `--epochs` defines the number of training epochs;
- `--embedding_size` defines the size of the latent/embedding space;
- `--mode` allows you to select between training or inference;
- `--number_nodes` number of nodes in graph;
- `--percent_edges` percentage of connections between nodes;
- `--plot_graph` boolean argument for plotting the graph;
- `--plot_pca` boolean argument to visualize the latent space generated using principal component analysis;
- `--save_model` enables the saving of parameters and weights for later inference;
- `--device` allows processing using GPU;
- `--most_similar` defines a range of the most similar nodes.

## Implementation aspects

- `Log Softmax` to transform logits into probabilities;
- `Cross-Entropy Loss` as a cost function.

## Simple Biased Random Walk Illustration

![](https://github.com/paulosantosneto/dl-bp/blob/main/gdl/node2vec/figures/biased_random_walk.png)

## References

[1] Grover, Aditya & Leskovec, Jure. (2016). node2vec: Scalable Feature Learning for Networks. KDD : proceedings. International Conference on Knowledge Discovery & Data Mining. 2016. 855-864. 10.1145/2939672.2939754.