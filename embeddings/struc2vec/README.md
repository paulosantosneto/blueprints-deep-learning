`Warning` This is my implementation of the article ['struc2vec: Learning Node Representations from Structural Identity'](https://arxiv.org/pdf/1704.03165.pdf). Therefore, it is not an official repository of it. Any code improvement recommendations are welcome.

**Complementary articles to struc2vec implemented**

- [Dynamic Time Warping](https://www.researchgate.net/publication/285279006_Dynamic_time_warping)

**To Train**
```
python3 main.py --mode train --seed 1 --graph barbell_graph --q 0.5 --epochs 100 --k 5 --number_walks 5 --walk_length 10 --plot_graph True --plot_pca True
```

**To Infer**
```
python3 main.py --mode inference --graph barbell__graph --model_load_path weights_barbell_graph_100 --target 0 --most_similar 5
```

## Simple Multilayer Graph Illustration

![](https://github.com/paulosantosneto/GNNs/blob/main/embeddings/struc2vec/figures/multilayer_graph.png)

Simplified representation of a random biased walk in the multilayer graph structure. The oriented dashed paths represent an example of a walk generated based on transitive probabilities proportional to the network structure.