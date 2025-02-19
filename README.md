# PMP-MetaDistill
this repo covers the implementation of the following paper:
PMP-MetaDistill: Enhancing Adversarial Robustness of Small Models via Adaptive Multi-Teacher Distillation with PMP-Guided Meta-Learning
![image](https://github.com/MaiEmily/map/blob/master/public/image/20190528145810708.png)
## Installation
This repo was tested with Python 3.10, Torch 1.10.0, torchvision 0.11.1 and CUDA 12.4 A100

## Running
Before distill the student, be sure to get a robust teacher model, you can use the file AT.py to get it.
and you also have a clean teacher model.
```
python AT.py  &&
python pmp_resnet_cifar10.py
```
## Citation
If you find this repository useful, please consider citing the following paper:
```
...

```

## Acknowledgement
This research was supported by the National Natural Science Foundation of China under Grant No.62302499
