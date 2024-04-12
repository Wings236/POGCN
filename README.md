# POGCN
This is the Pytorch implementation for our paper:

## Introduction
In this study, we utilize the partial order relation to model multi-behavior interactions, thus obtaining a multi-behavior partial order relation. We then use this relation to enhance both the graph convolutional neural network and BPR training, which we refer to as POG and POBPR respectively. The overall method is termed as POGCN.

## Enviroment Requirement
`pip install -r requirements.txt`

## Dataset
We provide the processed datasets: taobao.

## An example to run
run POGCN on taobao dataset:
  
`cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=8888 --dataset="taobao" --behaviors "['click', ['cart', 'fav'], 'buy']" --level 1.0 --sample_level 1.0 --topks="[20]" --recdim=64`



