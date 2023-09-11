`Warning` This is my implementation of the article ['DeepWalk: Online Learning of Social Representations'](https://arxiv.org/pdf/1403.6652.pdf). Therefore, it is not an official repository of it. Any code improvement recommendations are welcome.

To train the model:
```
python3 deepwalk.py --mode train --number_nodes 50 --edges 0.1
```

To infer:
```
python3 deepwalk.py --mode inference --model_load_path weights_50.pth --model_config_path parameters.json --node 0
```

## Simple Ilustration of Random Walk

![](https://github.com/paulosantosneto/GNNs/blob/main/graph_embeddings/deepwalk/random_walk.png)
