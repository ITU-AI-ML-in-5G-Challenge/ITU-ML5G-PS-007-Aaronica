import brevitas.nn as qnn
import torch.nn.utils.prune as prune
from torch import nn
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from train import *
from utils import logger, save_checkpoint
import os

def prune_api(model, th):
    parameters_to_prune = []
    for m in model.modules():
        if isinstance(m, qnn.QuantLinear) or isinstance(m, qnn.QuantConv1d):
            parameters_to_prune.append((m, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=th,
    )

def burn_prune(model):
    for module in model.modules():
        if isinstance(module, (qnn.QuantConv1d, qnn.QuantLinear)) and prune.is_pruned(module):
            prune.remove(module, 'weight')




def retrain(model, epoch, lr, T_0, data_loader_train, data_loader_test, iter_, gpu, weight_decay, out_folder):
    log_file = os.path.join(out_folder,'log.txt')
    criterion = nn.CrossEntropyLoss()
    if gpu != -1:
        criterion = criterion.cuda()
        model = model.cuda()
    if weight_decay != 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1)

    for epoch in tqdm(range(epoch), desc="Epochs"):
        loss_epoch = train(model, data_loader_train, optimizer, criterion, gpu)
        test_acc = test(model, data_loader_test, gpu)
        info_ = "Epoch %d: Training loss = %f, test accuracy = %f\n" % (epoch, np.mean(loss_epoch), test_acc)
        logger(info_, log_file)
        lr_scheduler.step()
        
        save_checkpoint(model, f"checkpoint_{epoch}_{iter_}", out_folder)
    return model, info_, test_acc
