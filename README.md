# AaronNet for ITU-ML5G-PS-007

## Table Of Contents
- [AaronNet for ITU-ML5G-PS-007](#aaronnet-for-itu-ml5g-ps-007)
  - [Table Of Contents](#table-of-contents)
  - [Metrics](#metrics)
  - [Usage:](#usage)
  - [Team AaronNet](#team-aaronnet)


## Metrics
1. AaronNet Pruned: 
- **Pruned Network Accuracy** `56.07`
- **Inference Cost** `0.01539`
2. AaronNet Base (used for pruning):
- **Pruned Network Accuracy** `58.48`
- **Inference Cost** `0.05722`
2. AaronNet Large:
- **Pruned Network Accuracy** `60.07`
- **Inference Cost** `0.04320`


## Usage:

1. Use `main.py` for training, pruning, and testing.
2. Use `evaluation.ipynb` to analyze the results of a checkpoint.
3. There are 3 checkpoints in the `checkpoints` folder:
   - `vgg.pth`: Baseline VGG Network
   - `main.pth`: AaronNet unpruned
   - `pruned.pth`: AaronNet pruned
   - `high_acc.pth`: AaronNet Large
  
python main.py --args
Argument List and help:
``` shell
('--bsize', default=1024, help='Batch Size')
('--lr', default=0.01, help='Learning Rate')
('--eps', default=50, help='Number of Training Epochs')
('--t0', default=5, help='LRScheduler parameter')
('--ret_ep', default=2, help='Retrain Epochs After Pruning - Multiplier of T_0')
('--st_snr', default=3, help='Starting SNR for training')
('--dset_path', default='', help='Path to RADIOML2018 HDF5 file')
('--checkpoint', default='submission_unpruned.pth', help='Starting Checkpoint')
('--w_dec', default=0.0, help='Weight Decay. Set to 0 to disable.')
('--nbits', default=8, help='Number of Quantization bits')
('--prn_list', default=[], help='Percentages of Pruning each Iteration. If not set, you have to use fixed percentage (prn_per))
('--prn_per', default=30, help='Fixed Pruning Percentage')
('--out_folder', default='result', help='Path to Save the Log File and Checkpoints.')
('--gpu_num', default=0, help='Select GPU. Set to -1 to run on CPU')
('--action', default=1, help='Set to 2 to Prune, Set to 1 to Train, Set to 0 to Test checkpoint.')
```

We will soon add a Docker container.

## Team Aaronica

- Mohammad Chegini: mohammad.chegini@hotmail.com
- Pouya Shiri: pouyashiri@uvic.ca
- Amirali Baniasadi: amiralib@uvic.ca
