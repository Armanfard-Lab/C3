# C3: Cross-instance guided Contrastive Clustering
PyTorch implementation of the paper "C3: Cross-instance guided Contrastive Clustering"

<center><img src="https://github.com/Armanfard-Lab/C3/blob/main/Figs/C3.jpg" alt="Overview" width="800" align="center"></center>

## Implementation

Please first downlaod the **`CIFAR_10_initial.zip`** from [this link](https://drive.google.com/file/d/1deqzG-eUztltgdQ0H2Y29N83_i9Tr_uN/view?usp=sharing) and put it in the same folder as **`main.py`** and then run the code.

## Citation

You can find the preprint of our paper on [arXiv](https://arxiv.org/abs/2211.07136).

Please cite our paper if you use the results or codes of our work.

```
@article{sadeghi2022c3,
  title={C3: Cross-instance guided Contrastive Clustering},
  author={Sadeghi, Mohammadreza and Hojjati, Hadi and Armanfard, Narges},
  journal={arXiv preprint arXiv:2211.07136},
  year={2022}
}
```

## Abstract

>Clustering is the task of gathering similar data samples into clusters without using any predefined labels. It has been widely studied in machine learning literature, and recent advancements in deep learning have revived interest in this field. Contrastive clustering (CC) models are a staple of deep clustering in which positive and negative pairs of each data instance are generated through data augmentation. CC models aim to learn a feature space where instance-level and cluster-level representations of positive pairs are grouped together. Despite improving the SOTA, these algorithms ignore the cross-instance patterns, which carry essential information for improving clustering performance. In this paper, we propose a novel contrastive clustering method, Cross-instance guided Contrastive Clustering (C3), that considers the cross-sample relationships to increase the number of positive pairs. In particular, we define a new loss function that identifies similar instances using the instance-level representation and encourages them to aggregate together. Extensive experimental evaluations show that our proposed method can outperform state-of-the-art algorithms on benchmark computer vision datasets: we improve the clustering accuracy by 6.8%, 2.8%, 4.9%, 1.3% and 0.4% on CIFAR-10, CIFAR-100, ImageNet-10, ImageNet-Dogs, and Tiny-ImageNet, respectively.
