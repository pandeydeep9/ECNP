# ECNP Code
This is the repository for the Code of Evidential Conditional Neural Processes <br />

##Appendix
The appendix and supplimentary works of the paper are provided in appendix.pdf file

## This code requires
* Python3
* Pytorch

## The datasets used are
* Synthetic Datasets: Sinusoidal regression and GP regression (generated in the code itself)
* Real world Image Completion Datasets: MNIST, CelebA and Cifar10

Download the datasets and organize the datasets with following structure

```buildoutcfg
datasets/
├── celeba
│   ├── test
│   ├── train
│   └── val
├── cifar10
│   ├── test
│   └── train
└── mnist
    ├── test
    └── train
```

Celeba has just 1 folder in both train and test consisting of all the task imags <br />
MNIST and Cifar10 should have 10 folders (numbered 0 - 9 corresponding to the 10 classes) in both train and test. <br />

# Running evidential models
```enp_run.py``` is used to run **ECNP-A** and **ECNP**<br />
For eg. to run 50-shot image-completion experiment with mnist for ECNP, use <br />
```python3 enp_run.py --dataset "mnist" -use_det "true" -use_lat "false" --max_context_points 50 --model_type "CNP"```

# Running baseline models (NP, CNP, ANP)
```np_baseline.py``` script can be used to run the baseline models <br />
For eg. to run 50 shot image completion experiment with mnist for CNP, use <br />
```python3 np_baseline.py --dataset "mnist" -use_det "true" -use_lat "false" --max_context_points 50 --model_type "CNP"```

Additionally, various hyperparameters and settings can be specified using the ```utilFiles/get_args.py``` file

## Reference codes used for the repository
- ANP anc CNP Official Codes - https://github.com/deepmind/neural-processes
