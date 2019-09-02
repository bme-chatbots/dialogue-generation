# Dialogue generation

*`Currently under development. Please feel free to contribute with either pull requests or by filling an issue.`*

Implementation of a neural dialogue generator model with pretrained XLNet  *[Yang et al. (2019)](https://arxiv.org/pdf/1906.08237.pdf)* and GPT2 architecture *[Radford et al. (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)* on daily dialog dataset *[Li et al. (2017)](https://arxiv.org/pdf/1710.03957.pdf)* (additional datasets are coming soon). Top-k sampling *[Fan et al. (2018)](https://arxiv.org/pdf/1904.09751.pdf)* and nucleus decoding *[Holtzman et al. (2019)](https://arxiv.org/pdf/1904.09751.pdf)* are available as decoding techniques.

## Usage

The model uses mixed precision training from nvidia/apex. Note that apex is not required and is only used if it is available. For installation guide of this module see the official [instructions](https://github.com/NVIDIA/apex).

To train the model clone this repository and install dependecies. The project uses cython to assemble batches for faster input pipeline.

```console
pip install -r requirements.txt

python setup.py build_ext --inplace

python run.py
```

The model can be trained with the following commands. Note that `<data_dir>` and `<model_dir>` are optional, as they are provided by default but you can also customize the location of model and data directories with those arguments. Training with different hyperparameters can be done by running the `train.py` script and passing the desired options as command line arguments.

```console
python -m src.train --model_name model_name
```

```console
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE src/train.py --model_name model_name
```

Available models are `xlnet-base-cased`, `xlnet-large-cased`, and `gpt2`, `gpt2-medium`, `gpt2-large`. Currently the only available dataset option is `dailydialog`. An interactive evaluation mode is available on the trained model by running the `interact` script.

```console
python -m src.interact --model_name model_name
```

Training the model is fast and easy on Google Colaboratory, which can be done from scratch by creating a new colab file in your Google Drive and running it with the following snippet. It is important to set the runtime type to GPU with a Tesla T4 unit as it can fully leverage mixed-precision training and is much faster than the older K80 version. You can check the current type by running the following line in a cell of your colab.

```bash
!nvidia-smi
```

Copy and run the following code in a cell of your colab file for installing the model.

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
!python dialogue-generation/run.py
```

## Results

**Coming soon**
