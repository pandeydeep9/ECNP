Appendix is provided as appendix.pdf

Dataset structure
place the datasets (MNIST, CelebA and Cifar10 datasets). The datasets structure for the datasets is as follows
MNIST
	- train
	- test

Cifar10
	- train
	- test
	
celeba
	- train
	- test

celeba has just 1 folder in both train and test consisting of all the task imags
MNIST and Cifar10 should have 10 folders (numbered 0 - 9 corresponding to the 10 classes) in both train and test.

#Running evidential models
enp_run.py is used to run ENP-A and ENP-C
For eg. to run 50 shot image completion experiment with mnist for ENP-C, use
python3 enp_run.py --dataset "mnist" -use_det "true" -use_lat "false" --max_context_points 50 --model_type "CNP"

enp_l_run.py is used to run ENP-L models
For eg. to run 50 shot image completion experiment with mnist for ENP-C, use
python3 enp_l_run.py --dataset "mnist" -use_det "false" -use_lat "true" --max_context_points 50 --model_type "CNP"

#Running baseline models (NP, CNP, ANP)
np_baseline.py script can be used to run the baseline models
For eg. to run 50 shot image completion experiment with mnist for CNP, use
python3 np_baseline.py --dataset "mnist" -use_det "true" -use_lat "false" --max_context_points 50 --model_type "CNP"

Additionally, various hyperparameters and settings can be specified using utilFiles/get_args.py file

