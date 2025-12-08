# Rigorously Characterizing Dynamics with Machine Learning

This code accompanies the paper [Rigorously Characterizing Dynamics with Machine Learning](https://arxiv.org/abs/2505.17302v1) by Marcio Gameiro, Brittany Gelb, and Konstantin Mischaikow.

We learn neural network approximations to the two-dimensional Leslie model, and then compute Morse graphs and Conley indices using the approximations.

# Dependencies 
This repository depends on [CMGDB](https://github.com/marciogameiro/CMGDB).

# Usage

## Data, scalers, and models
The datasets used for the paper are contained in ```data```.
The data scalers used for the paper are contained in ```scalers```.
The models used for the paper are contained in ```models_and_logs```.

## Baseline
Perform the baseline computations using ```baseline.py``` or this [Colab notebook](https://colab.research.google.com/drive/1WDTQrC_SYKvfRHFLYTfM6FJf78rD5SM-?usp=sharing).

## Data
Make new datasets using ```make_data.py```.
Compute data scalers using ```scale_data.py```.

## Config files
Change configuration files located in ```config```

## Train model
Train neural network models using ```train.py```.

## Compute Morse graph
Compute Morse graph and Conley indices from neural network model using ```morse_graph.py```
