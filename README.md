# dialogue generation

This repository contains training utilities for **`GPT-2`** architecture *[Radford et al. (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)* on currently two datasets: **`DailyDialog`** *[Li et al. (2017)](https://arxiv.org/pdf/1710.03957.pdf)* , **`PersonaChat`** *[Zhang et al. (2018)](https://arxiv.org/pdf/1801.07243.pdf)*. For more technical details of GPT-2 for dialogue-generation see its *[README.md](https://github.com/bme-chatbots/dialogue-generation/blob/version2/experiments/gpt2/README.md)*.

## installation

Installation is straight-forward with python virtualenv and pip by running the commands below. The main dependencies are *[transformers](https://github.com/huggingface/transformers)* for model implementations, *[datasets](https://github.com/huggingface/datasets)* for large memory-mapped dialogue data pipelines, *[hydra](https://github.com/facebookresearch/hydra)* for managing project configurations and *[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)* to avoid writing boilerplate code for training loops and model management.

```console
git clone https://github.com/Mrpatekful/dialogue-generation.git

cd dialogue-generation; pip install -r requirements.txt
```

## training

Training is performed by **`run_training.py`** scripts in the corresponding experiments subdirectories.

```console
cd dialogue-generation/experiments/gpt2; python run_training.py data.max_tokens=1500 pretrained_name=microsoft/DialoGPT-small
```

Resuming training of a model is also possible by setting the `checkpoint_file` to the desired `.ckpt` path.

```console
cd dialogue-generation/experiments/gpt2; python run_training.py checkpoint_file=.../last.ckpt data.max_tokens=1500 pretrained_name=microsoft/DialoGPT-small
```

Training configuration **`config.yaml`** file can be found next to the training script under the **`config`** directory. To override the default parameters see Hydra configuration framework *[project page](https://github.com/facebookresearch/hydra)*. 
 
```yaml
├── experiments
|   └── gpt2
|       ├── run_training.py
|       ├── config
|       |   └── config.yaml
|       ...
|       └── run_generation.py
├── scripts
```

## interaction

Interaction with your trained model is possible by running **`run_generation.py`** script and setting the `checkpoint_file` to the desired `.ckpt` path.

```console
export CHECKPOINT_FILE=dialogue-lm/experiments/gpt2/outputs/Y-m-d/H-M-S/last.ckpt
cd dialogue-lm/experiments/gpt2; python run_generation.py checkpoint_file=$CHECKPOINT_FILE generation.temperature=0.8
```
