`Warning` This is my implementation of the article ['The graph neural network model'](https://ro.uow.edu.au/cgi/viewcontent.cgi?article=10501&context=infopapers)[1] and the article ['Automatic generation of complementary descriptors with molecular graph networks'](https://pubmed.ncbi.nlm.nih.gov/16180893/)[2]. Therefore, it is not an official repository of it. Any code improvement recommendations are welcome.

**To Train**

```
python3 main.py --epochs 100 --hidden_layers 16 --dataset cora
```

**Arguments**

- `--epochs` defines the number of training epochs;
- `--mode` allows you to select between training or inference;
- `--device` allows processing using GPU;
- `--dataset` allows you to select between CORA or Zachary dataset;
- `--hidden_layers` number of units per hidden layer;
- `--weight_decay` decay rate for the Adam optimizer;
- `--learning_rate` learning rate for the Adam optimizer.

## Simple Ilustration of Message Passing


![](https://github.com/paulosantosneto/GNNs/blob/main/dl_based_methods/GNN/figures/message_passing.png)

## References

[1] Scarselli, M. Gori, A.C. Tsoi, M. Hagenbuchner, and G. Monfardini. The
graph neural network model. IEEE Trans. Neural Netw. Learn. Syst, 20(1):
61–80, 2009.

[2] Merkwirth and T. Lengauer. Automatic generation of complementary descriptors with molecular graph networks. J. Chem. Inf. Model, 45(5):1159–
1168, 2005.
