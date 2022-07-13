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

The following command will start training on a single GPU/CPU with `gpt2-medium` model on `PersonaChat`. `--name` is the name of the subdirectory in the model folder, where logs and checkpoints are saved.

```console
python -m src.train --model gpt2-medium --data personachat --name my_test_run
```

For distributed multi-gpu training the train script should be called like this.

```console
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS src/train.py --model gpt2
```

You can also use predefined configs by passing the path of the config json file as `--config` argument. These are available in `src/configs` folder and their training results can be seen below the results section.

```console
python -m src.train --config src/configs/xlnet-dailydialog.json
```

Training the model is fast and easy on *[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)* or *[Kaggle kernel](https://www.kaggle.com/kernels)*. It is important to set the runtime type to GPU with the new Tesla P100 or Tesla T4 unit as it can fully leverage mixed-precision training and is much faster than the older Tesla K80 version. You can check the current type by running `!nvidia-smi` in a cell of your colab.

*As a shortcut here is a complete [example gist](https://gist.github.com/Mrpatekful/94aa58038cdd221cfa83a7e7334f3835), which you can simply import to your Google Drive as a colaboratory file.*

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

The training and validation metrics are logged to Tensorboard, which can also be tracked in the colab file if the below code is run before the training cell.

```bash
%load_ext tensorboard
```

```bash
%tensorboard --logdir "dialogue-generation/model"
```

The model can be trained then by simply running the `train` script with the default flags. You can see all flags accepted by the `train.py` script by providing `-h` flag.

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

```console
python -m src.interact --model gpt2-medium --name my_test_run
```

```console
python -m src.interact --config src/configs/xlnet-dailydialog.json
```

## Customization

To train any model on your own dataset you simply have to subclass from `DialogDataset` and implement data generation from the raw files. Given a `train.txt`, `valid.txt` and `test.txt` placed in `data\<name of your data>`, where each turn in a dialog is in a new line and separate dialogs are divided by an extra empty line.

```yaml
├── data
|   ├── dailydialog
|   └── custom_dataset   # name of your data ( custom_dataset by default )
|       ├── train.txt    # the correctly formated train, valid and test files
|       ├── valid.txt
|       └── test.txt
├── src
```

```text
Hello how are you?
Hi I'm fine thanks. And you?
Me too thanks for asking.

Hi my name Peter.
Nice to meet you I am Eric.
```

An example custom dataset class named `CustomDataset` is implemented in [`data.py`](https://github.com/bme-chatbots/dialogue-generation/blob/05a140b39bf8fb5b12ebb5f7f9f77e56f38ce752/src/data.py#L954) that reads a dataset with these properties.

## Results

The below dialog is sampled from `gpt2` trained with default parameters for 7 epochs ( 149051 steps ) and `--max_hist 3` on `PersonaChat` dataset.

```text
User: hi

Bot: greetings, i'm gerald.

User: how are you ?

Bot: good, thanks. getting ready for another hike, i love hiking!

User: oh realy? where are you going?

Bot: pikes peak. i am curently exploring famous mountains when i have time.

User: that is cool are you going alone?

Bot: my brother is in the us, we go hiking with him every month.

User: how old are you by the way?

Bot: a little under 46 years old

User: are you in a relationship?

Bot: yes i am married, 3 boys, i am pregnant again. i am too busy to focus
```

----

The below dialog is sampled from `gpt2` trained with default parameters for 8 epochs ( 48824 steps ) and `--max_hist 3` on `DailyDialog` dataset.

```text
User: hi how are you ?

Bot: fine. i got a promotion at work earlier this week.

User: where do you work ?

Bot: I work at the polling place.

User: where is that ?

Bot: I live in the dorms across the street.

User: do you like it ?

Bot: it's great. I heard it is far from here.

User: are you a student ?

Bot: yes. I am a student too.
```
