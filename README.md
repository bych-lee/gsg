# Implicit Contrastive Representation Learning with Guided Stop-gradient

This is an implementation of [our paper](https://neurips.cc/virtual/2023/poster/71356) in NeurIPS 2023.

### Preparation

Download the ImageNet dataset (https://www.image-net.org/download.php). <br>
Install PyTorch (https://pytorch.org/). <br>
Install apex (https://github.com/NVIDIA/apex) for LARS optimizer needed in linear evaluation.

### Pre-training

```
python main_pretrain.py --model simsiam_gsg --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 --fix-pred-lr \
  --save-path [path to a folder where checkpoints will be saved] \
  [your imagenet-folder with train and val folders]
```

### k-nearest Neighbors

```
python main_knn.py --model simsiam_gsg --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [path to a pre-trained checkpoint] \
  [your imagenet-folder with train and val folders]
```

### Linear Evaluation

```
python main_lincls.py --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [path to a pre-trained checkpoint] \
  [your imagenet-folder with train and val folders]
```

This code is based on Exploring Simple Siamese Representation Learning by Xinlei Chen and Kaiming He: https://github.com/facebookresearch/simsiam/blob/main/LICENSE.