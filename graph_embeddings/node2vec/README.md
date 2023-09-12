`Warning` This is my implementation of the article ['node2vec: Scalable Feature Learning for Networks'](https://arxiv.org/pdf/1607.00653.pdf). Therefore, it is not an official repository of it. Any code improvement recommendations are welcome.

To train the model:
```
python3 node2vec.py --mode train --number_nodes 50 --percent_edges 0.1 --p 0.2 --q 0.8
```

To infer:
```
python3 node2vec.py --mode inference --model_load_path weights_50.pth --model_config_path parameters.json --node 0
```
## Simple Ilustration of Biased Random Walk

![](https://github.com/paulosantosneto/GNNs/blob/main/graph_embeddings/node2vec/biased_random_walk.png)

