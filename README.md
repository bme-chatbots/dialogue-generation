# Dialogue generation

Implementation of a neural dialogue generator model with pretrained **`XLNet`**  *[Yang et al. (2019)](https://arxiv.org/pdf/1906.08237.pdf)* and **`GPT2`** architecture *[Radford et al. (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)* on currently three datasets: **`DailyDialog`** *[Li et al. (2017)](https://arxiv.org/pdf/1710.03957.pdf)* , **`PersonaChat`** *[Zhang et al. (2018)](https://arxiv.org/pdf/1801.07243.pdf)* and the new **`TopicalChat`** *[Gopalakrishnan et al. (2019)](https://m.media-amazon.com/images/G/01/amazon.jobs/3079_Paper._CB1565131710_.pdf)* from [Alexa Prize Socialbot Grand Challenge 3](https://developer.amazon.com/blogs/alexa/post/30dc5515-3b9f-4ec2-8f2a-ac98254625c6/topical-chat-dataset-helps-researchers-address-hard-challenges-in-natural-conversation). Top-k sampling *[Fan et al. (2018)](https://arxiv.org/pdf/1904.09751.pdf)* and nucleus decoding *[Holtzman et al. (2019)](https://arxiv.org/pdf/1904.09751.pdf)* are available as decoding techniques. The training objective is autoregressive language modeling on the utterances and dialogue histories.

## Installation

The model can leverage mixed precision training from nvidia/apex. Note that apex is not required and is only used if it is available. For installation guide see the official [instructions](https://github.com/NVIDIA/apex). Using this module is not useful for all GPUs ( only Volta and Turing ) and you should check in prior if your instance supports mixed precision training.

To train the model clone this repository and install dependecies. The project uses cython to assemble batches for faster input pipeline. It also preferred to use a python virtualenv.

```console
git clone https://github.com/bme-chatbots/dialogue-generation.git

cd dialogue-generation

pip install -r requirements.txt

python setup.py build_ext --inplace
```

## Training

The model can be trained with the following commands. Note that `--data_dir` and `--model_dir` are optional, as they are provided by default but you can also customize the location of model and data directories with those arguments. The exact path of the model is `<model_dir>/<model>/<name>` where the name subdirectory is given by the `--name` argument ( `DD:MM:YY-hh-mm-ss` by default ) contains the logs and training checkpoints for a particular run, while `<model>` contains the pretrained initial checkpoint of the model. This is useful if one would like to train a model on several datasets in a consecutive manner, which can be done by mainting the same `--name` argument and changing the `--data` parameter.

```console
python -m src.train --model gpt2-medium --data personachat --name my_test_run
```

For distributed multi-gpu training the train script should be called like this.

```console
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS src/train.py --model gpt2
```

You can also use predefined configs by passing the path of the config json file as `--config` argument. These are available in `src/configs` folder and their training results can be seen below the **results** section.

```console
python -m src.train --config src/configs/xlnet-dailydialog.json
```

Available models are **`xlnet-base-cased`**, **`xlnet-large-cased`**, and **`distilgpt2`** **`gpt2`**, **`gpt2-medium`**, **`gpt2-large`**, **`gpt2-xl`**. Currently the available dataset options are **`dailydialog`**, **`personachat`**, **`topicalchat`** but you can easily extend them by adding your own. Example to create your own dataset can be seen below.

Training the model is fast and easy on *[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)* or *[Kaggle kernel](https://www.kaggle.com/kernels)*, which can be done from scratch by creating a new colab file in your Google Drive and running it with the following snippet. It is important to set the runtime type to GPU with the new Tesla P100 or Tesla T4 unit as it can fully leverage mixed-precision training and is much faster than the older Tesla K80 version. You can check the current type by running `!nvidia-smi` in a cell of your colab.

Copy and run the following code in a cell of your colab *( or Kaggle kernel )* file to install the model. If you use Kaggle kernel you also have to enable internet access.

```bash
!git clone https://github.com/bme-chatbots/dialogue-generation.git
!python -m pip install --upgrade pip

# installing apex is optional and is only useful if Colab's Tesla P100 or T4 is used
# !git clone https://github.com/NVIDIA/apex
# !cd apex; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# building the cython code and installing the required packages
!cd dialogue-generation; pip install -r requirements.txt; python setup.py build_ext --inplace
```

The training loss and accuracy is logged with TensorboardX, which can also be tracked in the colab file if the below code is run before the training cell.

```bash
%load_ext tensorboard
```

```bash
%tensorboard --logdir "dialogue-generation/model"
```

The model can be trained then by simply running the `train` script with the default flags. Flags for the training scripts with the default values are the following.

**`train`**

- `--config` *Path of the config file that contains flags. ( default: None )*

- `--max_epochs` *Maximum number of epochs for training. ( default: 25 )*

- `--no_cuda` *Do not use GPU for training. ( default: False )*

- `--fp16` *Use half-precision training. ( default: False )*

- `--lr` *Learning rate for the optimizer. ( default: 1e-5 )*

- `--batch_size` *Batch size for training and validation. ( default: 64 )*

- `--patience` *Patience value for early stopping. ( default: 5 )*

- `--schedule` *Type of learning rate scheduling to use. ( default: noam )*

- `--warmup_steps` *Number of warmup steps. ( default: 0.1 )*

- `--total_steps` *Number of total optimization steps. ( default: 1000000 )*

- `--grad_accum_steps` *Number of steps for grad accum. ( default: 2 )*

- `--notebook` *Render progressbar in notebook mode. ( default: False )*

- `--clip_grad` *Value of gradient clipping. ( default: None )*

- `--seed` *Random seed for the training. ( default: None )*

**`model`**

- `--model` *Name of model for training. ( default: xlnet-base-cased )*

- `--grad_ckpt` *Use gradient checkpointing. ( default: False )*

- `--name` *Name of the current training session. ( default: %y.%m.%d-%H:%M:%S )*

- `--model_dir` *Path of the model root directory. ( default: `<PROJECT_DIR>`/model )*

**`data`**

- `--data` *Name of the dataset to use for training. ( default: dailydialog )*

- `--data_dir` *Path of the root data directory. ( default: `<PROJECT_DIR>`/data )*

- `--download_dir` *Path of download root directory. ( default: `<PROJECT_DIR>`/data )*

- `--file_size` *Max utterances stored in a single file. ( default: 100000 )*

- `--max_hist` *Num utterances in the history. ( default: 2 )*

- `--force_rebuild` *Recreate the data even if it exists. ( default: False )*

- `--max_len` *Maximum length of an utterance. ( default: 50 )*

```bash
!cd dialogue-generation; python -m src.train
```

After training the model can be downloaded by setting the download link in the following snippet to the one logged by the script after evaluation. ( `Saving model to dialogue-generation/src/../model/gpt2/19.11.03-12:59:47/model.pt` )

```python
from IPython.display import FileLink

# note that in case of kaggle kernel you have to give path
# relative to your working directory
FileLink(r'dialogue-generation/src/../model/gpt2/19.11.03-12:59:47/model.pt')
```

## Interaction

An interactive evaluation mode is available on the trained model by running the `interact` script and providing the path of the trained model with `--model_file`. You can also provide the `--config` file or just simply give the same `--model` and `--name` argument, which was used during training.

**`interact`**

- `--model_file` *Name of the model file; ( default: None )*

- `--ckpt_name` *If loading model by `--name` and not `--model_file` use the `last` or `best` version ( default: `last` )*

- `--method` *Decoding method for interaction ( default: nucleus )*

- `--no_cuda` *Do not use GPU for training. ( default: False )*

- `--top_p` *Top-p parameter for nucleus decoding. ( default: 0.9 )*

- `--top_p` *Top-k parameter for topk sampling. ( default: 100 )*

- `--seed` *Random seed for the training. ( default: None )*

**`model`** and **`data`** arguments are the same.

```console
python -m src.interact --model gpt2-medium --name my_test_run
```

```console
python -m src.interact --config src/configs/xlnet-dailydialog.json
```

## Customization

To train any model on your own dataset you simply have to subclass from `DialogDataset` and implement data generation from the raw files. Given a `train.txt`, `valid.txt` and `test.txt` placed in `data\<name of your data>`, where each turn in a dialog is in a new line and separate dialogs are divided by an extra empty line.

```text
Hello how are you?
Hi I'm fine thanks. And you?
Me too thanks for asking.

Hi my name Peter.
Nice to meet you I am Eric.
```

An example custom dataset class named `CustomDataset` is implemented in [`data.py`](https://github.com/bme-chatbots/dialogue-generation/blob/05a140b39bf8fb5b12ebb5f7f9f77e56f38ce752/src/data.py#L954) that reads a dataset with these properties.

## Results

These results are from `xlnet-base-cased` trained with default parameters for 14 epochs and **`--max_hist 1`**. More evaluation results are coming soon.

```text
what is your name ?
Susan Ann.


how old are you ?
about three and a half, right?


where do you live ?
in California.


what is your favourite colour ?
I like red, both in both of them!


do you have a favourite movie ?
sure. It was quite popular among children.


what is the title of this movie ?
it's called'Is it a new film '


you are not funny
I am not joking, but you deserve to change your mind.
```
