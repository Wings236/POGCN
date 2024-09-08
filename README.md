# Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks(POGCN)
<a href="https://github.com/Wings236/POGCN/"><img src="https://img.shields.io/badge/Project-Web-Green"></a>
<a href="https://arxiv.org/pdf/2402.07659"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 

## Introduction
In this work, we utilize the partial order relation to model the multi-behavior interactions of users and items, such as *click, cart, favor, like, share, buy* and so on. 
Then, we use this relationship to improve the Graph Neural Network and the Bayesian Personalized Ranking(BPR), which are called Partial Order Graph(**POG**) and Partial Order BPR(**POBPR**) respectively.
Finally, the whole method is called as **POGCN**.

## Enviroment Requirement
`pip install -r requirements.txt`

## Dataset
We provide the processed datasets: taobao(click, favor, cart and buy).

## An example to run
run POGCN on taobao dataset:

`cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=8888 --dataset="taobao" --behaviors "['click', ['fav', 'cart'], 'buy']" --level 1.0 --sample_level 1.0 --topks="[20]" --recdim=64`

## Citation
```bibtex
@inproceedings{zhang2024multi,
  title={Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks},
  author={Zhang, Yijie and Bei, Yuanchen and Chen, Hao and Shen, Qijie and Yuan, Zheng and Gong, Huan and Wang, Senzhang and Huang, Feiran and Huang, Xiao},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6257--6268},
  year={2024}
}
