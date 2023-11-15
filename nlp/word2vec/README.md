# Word2vec

`Warning` This is my implementation of the articles ['Efficient Estimation of Word Representations in Vector Space'](https://arxiv.org/pdf/1301.3781.pdf)[1] and ['Distributed Representations of Words and Phrases
and their Compositionality'](https://arxiv.org/pdf/1310.4546.pdf)[2]. Therefore, it is not an official repository of it. Any code improvement recommendations are welcome.

Complementary readings: [CS224N Winter 2019](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/readings/cs224n-2019-notes01-wordvecs1.pdf)

## Architecture

![](https://github.com/paulosantosneto/NLP/blob/main/word2vec/arch.jpg)

To train the model:
```
python3 skipgram.py --mode train --path_file train.txt --epochs 50 --embedding_size 100
```
To infer:
```
python3 skipgram.py --mode inference --model_load_path weights_skipgram_50.pth
```

## References

[1] Mikolov, Tomas & Chen, Kai & Corrado, G.s & Dean, Jeffrey. (2013). Efficient Estimation of Word Representations in Vector Space. Proceedings of Workshop at ICLR. 2013.

[2] Mikolov, Tomas & Sutskever, Ilya & Chen, Kai & Corrado, G.s & Dean, Jeffrey. (2013). Distributed Representations of Words and Phrases and their Compositionality. Advances in Neural Information Processing Systems. 26.

