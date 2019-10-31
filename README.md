# Dialogue generation

Implementation of a neural dialogue generator model with pretrained **XLNet**  *[Yang et al. (2019)](https://arxiv.org/pdf/1906.08237.pdf)* and **GPT2** architecture *[Radford et al. (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)* on currently three datasets: **DailyDialog** *[Li et al. (2017)](https://arxiv.org/pdf/1710.03957.pdf)* , **PersonaChat** *[Zhang et al. (2018)](https://arxiv.org/pdf/1801.07243.pdf)* and the new **TopicalChat** *[Gopalakrishnan et al. (2019)](https://m.media-amazon.com/images/G/01/amazon.jobs/3079_Paper._CB1565131710_.pdf)* from [Alexa Prize Socialbot Grand Challenge 3](https://developer.amazon.com/blogs/alexa/post/30dc5515-3b9f-4ec2-8f2a-ac98254625c6/topical-chat-dataset-helps-researchers-address-hard-challenges-in-natural-conversation). Top-k sampling *[Fan et al. (2018)](https://arxiv.org/pdf/1904.09751.pdf)* and nucleus decoding *[Holtzman et al. (2019)](https://arxiv.org/pdf/1904.09751.pdf)* are available as decoding techniques.

## Usage

### Installation

The model uses mixed precision training from nvidia/apex. Note that apex is not required and is only used if it is available. For installation guide of this module see the official [instructions](https://github.com/NVIDIA/apex).

To train the model clone this repository and install dependecies. The project uses cython to assemble batches for faster input pipeline. It also preferred to use a python virtualenv.

```console
git clone https://github.com/bme-chatbots/dialogue-generation.git

cd dialogue-generation

pip install -r requirements.txt

python setup.py build_ext --inplace
```

### Training

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

Available models are **`xlnet-base-cased`**, **`xlnet-large-cased`**, and **`distilgpt2`** **`gpt2`**, **`gpt2-medium`**, **`gpt2-large`**. Currently the available dataset options are **`dailydialog`**, **`personachat`**, **`topicalchat`** but you can easily extend them by adding your own. Example to create your own dataset can be seen below.

Training the model is fast and easy on *[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)* or *[Kaggle kernel](https://www.kaggle.com/kernels)*, which can be done from scratch by creating a new colab file in your Google Drive and running it with the following snippet. It is important to set the runtime type to GPU with a Tesla T4 unit as it can fully leverage mixed-precision training and is much faster than the older K80 version. You can check the current type by running `!nvidia-smi` in a cell of your colab. Note that Kaggle kernel's GPUs are faster than K80 and also don't go idle.

Copy and run the following code in a cell of your colab *( or Kaggle kernel )* file to install the model. If you use Kaggle kernel you also have to enable internet access.

```bash
!git clone https://github.com/bme-chatbots/dialogue-generation.git
!python -m pip install --upgrade pip

# installing apex
!git clone https://github.com/NVIDIA/apex
!cd apex; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# building the cython code
!cd dialogue-generation; python setup.py build_ext --inplace

# installing the required packages
!cd dialogue-generation; pip install -r requirements.txt
```

The training loss and accuracy is logged with TensorboardX, which can also be tracked in the colab file if the below code is run before the training cell.

```bash
%load_ext tensorboard
```

```bash
%tensorboard --logdir "dialogue-generation/model"
```

The model can be trained then by simply running the `run.sh` script with the default parameters.

```bash
!cd dialogue-generation; python -m src.train
```

### Interaction

An interactive evaluation mode is available on the trained model by running the `interact` script and providing the path of the trained model with `--model_file`. You can also provide the `--config` file or just simply give the same `--model` and `--name` argument, which was used during training.

```console
python -m src.interact --model gpt2-medium --name my_test_run
```

```console
python -m src.interact --config src/configs/xlnet-dailydialog.json
```

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
