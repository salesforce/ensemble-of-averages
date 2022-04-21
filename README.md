# Ensemble of Averages: Improving Model Selection and Boosting Performance in Domain Generalization

Official PyTorch implementation of [Ensemble of Averages](https://arxiv.org/pdf/2110.10832)

This repository is built upon the [DomainBed](https://github.com/facebookresearch/DomainBed) repository by FAIR.

## Environment:
```
	Python: 3.6.8
	PyTorch: 1.9.0+cu111
	Torchvision: 0.10.0+cu111
	CUDA: 11.1
	CUDNN: 8005
	NumPy: 1.19.5
	PIL: 8.4.0
  ```
  
## Run Commands

### Simple Moving Average (SMA):
First we train the models with our SMA protocol.
  
All experiments below use the ImageNet pre-trained ResNet-50 architecture, specified as `resnet50` in the command using the `arch` hyper-parameter. Other supported options include `resnext50_swsl` and `regnety_16gf` corresponding to weakly-supervised pre-trained models `ResNeXt-50 32x4d` from [Yalniz et al](https://arxiv.org/pdf/1905.00546), and `RegNetY-GF` from [Singh et al](https://arxiv.org/pdf/2201.08371.pdf) respectively.
  
PACS:
```
python -m domainbed.scripts.sweep launch --data_dir path/to/data --output_dir erm-sma_resnet50/pacs --command_launcher multi_gpu --algorithms ERM_SMA --datasets PACS --n_hparams 3 --n_trials 2 --single_test_envs --hparams '{"arch": "resnet50"}'
```

VLCS:
```
python -m domainbed.scripts.sweep launch --data_dir path/to/data --output_dir erm-sma_resnet50/vlcs --command_launcher multi_gpu --algorithms ERM_SMA --datasets VLCS --n_hparams 3 --n_trials 2 --single_test_envs --hparams '{"arch": "resnet50"}'
```

OfficeHome:
```
python -m domainbed.scripts.sweep launch --data_dir path/to/data --output_dir erm-sma_resnet50/officehome --command_launcher multi_gpu --algorithms ERM_SMA --datasets OfficeHome --n_hparams 3 --n_trials 2 --single_test_envs --hparams '{"arch": "resnet50"}'
```

TerraIncognita:
```
python -m domainbed.scripts.sweep launch --data_dir path/to/data --output_dir erm-sma_resnet50/terra --command_launcher multi_gpu --algorithms ERM_SMA --datasets TerraIncognita --n_hparams 3 --n_trials 2 --single_test_envs --hparams '{"arch": "resnet50"}'
```

DomainNet (notice that the number of steps is set to `15000` for this dataset following [SWAD](https://arxiv.org/abs/2102.08604)):
```
python -m domainbed.scripts.sweep launch --data_dir path/to/data --output_dir erm-sma_resnet50/domainnet --command_launcher multi_gpu --algorithms ERM_SMA --datasets DomainNet --n_hparams 3 --n_trials 2 --single_test_envs --hparams '{"arch": "resnet50"}' --steps 15000
```

### Ensemble of Averages (EoA)
We now use the best SMA models saved from the above runs (using in-domain validation accuracy based early stopping) in an ensemble, that we call EoA since these ensembles contain moving average models.

TIP: Use larger values of `num_workers` and `batch_size` for faster runtime.

```
python -m domainbed.EoA --data_dir /export/share/datasets/vision/domain-gen/PACS/kfold/ --dataset PACS --output_dir erm-sma_resnet50/pacs --hparams '{"num_workers": 1, "batch_size": 128}'
```


