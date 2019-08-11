# Dialogue generation

Implementation of a neural dialogue generator model with pretrained XLNet architecture *[Yang et al. (2019)](https://arxiv.org/pdf/1906.08237.pdf)* on daily dialog dataset *[Li et al. (2017)](https://arxiv.org/pdf/1710.03957.pdf)*. Top-k sampling *[Fan et al. (2018)](https://arxiv.org/pdf/1904.09751.pdf)* and nucleus decoding *[Holtzman et al. (2019)](https://arxiv.org/pdf/1904.09751.pdf)* are available as decoding techniques. Currently working on fine-tuning the input tensors (role embeddings) for the XLNet model.

## Usage

The model uses mixed precision training from nvidia/apex. Note that apex is not required and is only used if it is available. For installation guide of this module see the official [instructions](https://github.com/NVIDIA/apex).

The model can be trained with the following command. Note that `<data_dir>` and `<model_dir>` are optional, as they are provided by default. Training with different hyperparameters can be done by running the `train.py` script and passing the desired options as command line arguments.

```console
./run.sh "train" "<data_dir>" "<model_dir>"
```

An interactive evaluation mode is available on the trained model by switching the `train` to the `eval` flag.

```console
./run.sh "eval" "<data_dir>" "<model_dir>"
```

Training the model is fast and easy on Google Colaboratory, which can be done from scratch by creating a new colab file in your Google Drive and running it with the following snippet. It is important to set the runtime type to GPU with a Tesla T4 unit as it can fully leverage mixed-precision training and is much faster than the older K80 version. You can check the current type by running the following line in a cell of your colab.

```bash
!nvidia-smi
```

Copy and run the following code in a cell of your colab file for installing and training the model.

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

!./dialogue-generation/run.sh "train" "."
```

## Results

**Coming soon**
