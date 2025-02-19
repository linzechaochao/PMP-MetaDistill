# PMP-MetaDistill
this repo covers the implementation of the following paper:
PMP-MetaDistill: Adaptive Multi-Teacher Distillation with PMP-Guided Meta-Learning Framework

## Installation
This repo was tested with Python 3.10, Torch 1.10.0, torchvision 0.11.1 and CUDA 12.4 A100

## Running
Before distill the student, be sure to get a roubust teacher model, you can use the file AT.py to get it.
```
python AT.py  &&
python  pmp_resnet_cifar10.py
```
## Citation
If you find this repository useful, please consider citing the following paper:
```
...

```

## Acknowledgement
This research was supported by the National Natural Science Foundation of China under Grant No.62302499
