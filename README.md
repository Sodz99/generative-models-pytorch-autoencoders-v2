# Generative Models Using PyTorch Autoencoders

To build Generative Models by using Autoencoders in PyTorch

## Overview
Many of us are aware of the Discriminative models, which generally focus on predicting the data labels. In contrast, Generative Models help to generate the output, which is very similar to the input data. For example, we have some data related to cars that can be used to classify the cars, but Generative models can learn the patterns in the data and generate car features completely different than the input data. In this project, we start by introducing Generative Models. The PyTorch framework is used to build Autoencoders on the MNIST dataset. Finally, we learn how to use Autoencoders as Generative Models followed by generating new images of digits by using the Generative Model.

## Approach
1. Introduction to Generative Models
2. Introduction to Autoencoders
3. Buiding Autoencoders on PyTorch
4. Model training on Google Colab
5. Building Generative model by using Autoencoders

Modular code overview

Once you unzip the modular_code.zip file, you can find the following folders within it.
1. data
2. model
3. run.py
4. train.py
5. requirements.txt
1. The data folder contains data_utils.py file which is used to download and transform the data. It will download and store the data in respective folder
2. The model folder contains autoencoder_decoder_model.py and the colab notebook
3. The train.py files contains the code for model training. The run.py is main file in which all functions are called
4. The requirements.txt file has all the required libraries with respective versions. Kindly install the file by using the command pip install -r requirements.txt Note: please use cuda versions of torch if CUDA is available

## Tech stack
➔ Language: Python ➔ Libraries: torch, torchvision, torchinfo, numpy, matplotlib

## How to run
```bash
pip install -r modular_code/requirements.txt
python modular_code/run.py
python modular_code/train.py
```
Notebooks: [`auto_encoder_colab.ipynb`](modular_code/model/auto_encoder_colab.ipynb)

## Project structure
```
.gitignore
README.md
modular_code/
  model/
    auto_encoder_colab.ipynb
    autoencoder_decoder_model.py
  requirements.txt
  run.py
  train.py
```

## Credits
Based on a ProjectPro guided project.
