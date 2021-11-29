import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import copy
import os
from datetime import datetime
import argparse
from network import AaronNet
from utils import *
from dataset_loader import *
from prune import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aaronica RF Classifier")
    parser.add_argument('--bsize', default=1024, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate')
    parser.add_argument('--eps', default=50, type=int, help='Number of Training Epochs')
    parser.add_argument('--t0', default=5, type=int, help='LRScheduler parameter')
    parser.add_argument('--ret_ep', default=2, type=int, help='Retrain Epochs After Pruning - Multiplier of T_0')
    parser.add_argument('--st_snr', default=3, type=int, help='Starting SNR for training')
    parser.add_argument('--dset_path', default='', type=str,
                        help='Path to RADIOML2018 HDF5 file')
    parser.add_argument('--checkpoint', default='submission_unpruned.pth', type=str, help='Starting Checkpoint')
    parser.add_argument('--w_dec', default=0.0, type=float, help='Weight Decay. Set to 0 to disable.')
    parser.add_argument('--nbits', default=8, type=int, help='Number of Quantization bits')
    parser.add_argument("--prn_list", nargs="+", default=[], help='Percentages of Pruning each Iteration. If not set, '
                                                                  'you have to use fixed percentage (prn_per)')
    parser.add_argument('--prn_per', default=30, type=int, help='Fixed Pruning Percentage')
    parser.add_argument('--out_folder', default='result', type=str, help='Path to Save the Log File and Checkpoints.')
    parser.add_argument('--gpu_num', default=0, type=int, help='Select GPU. Set to -1 to run on CPU')
    parser.add_argument('--action', default=1, type=int, help='Set to 2 to Prune, Set to 1 to Train, Set to 0 to Test checkpoint.')
    
    args = parser.parse_args()

    # Prepare Pruning Strategy
    pruning_percentages = [float(x)/100.0 for x in args.prn_list]
    final_prune_per = args.prn_per
    if len(pruning_percentages) == 0:
        prune_percentage = args.prn_per / 100
        prune_iters = int(np.log(0.01) / np.log(1 - prune_percentage))
        pruning_percentages = [prune_percentage] * prune_iters
    else:
        final_prune_per = int(pruning_percentages[0]*100)

    os.makedirs(os.path.join(args.out_folder,'checkpoints'), exist_ok=True)
    log_file = os.path.join(args.out_folder,'log.txt')
    logger(str(args), log_file)


    # Set GPU/CPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_num)
        logger(f"Using {torch.cuda.get_device_name(args.gpu_num)}", log_file)
    else:
        gpu = None
        logger("Using CPU", log_file)


    # Loading dataset
    dataset_path = os.path.join(args.dset_path, '2018.01','GOLD_XYZ_OSC.0001_1024.hdf5')
    dataset_train = radioml_18_dataset(dataset_path, args.st_snr)
    dataset_test = radioml_18_dataset(dataset_path)
    data_loader_train = DataLoader(dataset_train, batch_size=args.bsize, sampler=dataset_train.train_sampler)
    data_loader_test = DataLoader(dataset_test, batch_size=args.bsize, sampler=dataset_test.test_sampler)


    # Setting seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)


    config = {
        "conv_layers": [
            #filters, ksize, padding, w_bits, a_bits, pool, bias
            [48, 3, 1, 6, 6, 2, True],
            [48, 3, 1, 6, 6, 2, True],
            [48, 3, 1, 6, 6, 2, True],
            [48, 3, 1, 6, 6, 2, True],
            [48, 3, 1, 6, 6, 2, True],
            [48, 3, 1, 6, 6, 2, True],
        ],
        "linear_layers": [48, 6, 6, True]
    }
    model = AaronNet(config, args.nbits)


    # prune
    if args.action == 2:
        model = load_checkpoint(model, args.checkpoint, log_file, args.gpu_num)

        logger("PRUNING STARTED!", log_file)
        # Pruning Loop
        for iter_ in range(prune_iters):
            prp = pruning_percentages[iter_]
            msg = f"\n\nIteration {iter_} is started at {str(datetime.now())} with {prp}\n\n"
            logger(msg, log_file)
            prune_api(model, prp)
            T_mult = 1
            for retrain_iter in range(args.ret_ep):
                epochs_ = T_mult * args.t0
                model, info_, t_acc = retrain(model, epochs_, args.lr, args.t0 * T_mult, data_loader_train, data_loader_test, iter_, args.gpu_num, args.w_dec, args.out_folder)
                if t_acc > 0.56:
                    break
                elif t_acc < 0.56:
                    T_mult = T_mult + 1
            pruned_model = copy.deepcopy(model)
            burn_prune(pruned_model)
            pruned_model_cost = calculate_cost(pruned_model)
            logger(f"BURN READY: Pruning Iter {iter_}\n{info_}\nIter {iter_} is done at {str(datetime.now())} with {prp}!\nCost is {pruned_model_cost}",log_file)
            save_checkpoint(pruned_model,f"Checkpoint_Burn_{iter_}", args.out_folder)
            logger(f"\nIter {iter_} is done at {str(datetime.now())} with {prp}!\nCost is: {pruned_model_cost}\n", log_file)
    
    # Train
    elif args.action == 1:
        criterion = nn.CrossEntropyLoss()
        if args.gpu_num != -1:
            criterion = criterion.cuda()
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1)

        running_loss = []
        running_test_acc = []
        logger("TRAINING STARTED!", log_file)
        for epoch in tqdm(range(args.eps), desc="Epochs"):
            loss_epoch = train(model, data_loader_train, optimizer, criterion, args.gpu_num)
            test_acc = test(model, data_loader_test, args.gpu_num)
            msg = "Epoch %d: Training loss = %f, test accuracy = %f" % (epoch+20, np.mean(loss_epoch), test_acc)
            logger(msg, log_file)
            running_loss.append(loss_epoch)
            running_test_acc.append(test_acc)
            lr_scheduler.step()
            save_checkpoint(model, f"training_checkpoint_{epoch}", args.out_folder)
    #Test
    else:
        model = load_checkpoint(model, args.checkpoint, log_file, args.gpu_num)
        test_acc = test(model, data_loader_test, args.gpu_num)
        print(f"ACC = {test_acc}")
        calculate_cost(model)