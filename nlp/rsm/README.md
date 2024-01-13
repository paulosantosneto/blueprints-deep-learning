

**To Train**
```
python3 main.py --mode train --sequence_model rnn --path_file ./data/small_dataset.txt --seed 123 --epochs 50 --sequence_length 10 --dimension 128 --plot_loss --verbose
```

**Arguments**

- `--epochs` defines the number of training epochs;
- `--mode` allows you to select between training or inference;
- `--sequence_model` allows you to select between three types of models: lstm, rnn or gru;
- `--device` allows processing using GPU;
- `--path_file` allows you to select your raw dataset (txt format);
- `--dimension` number of units per hidden layer;
- `--learning_rate` learning rate for the Adam optimizer.

**To Infer**
```
python3 main.py --mode inference --model_load_path best_model_50.pth --start_sequence artificial --model_config_path parameters.json
```

### Resources

https://www.cs.toronto.edu/~graves/preprint.pdf

https://arxiv.org/pdf/1402.1128.pdf

http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf