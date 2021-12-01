import torch
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from finn.util.inference_cost import inference_cost
import json, os
from matplotlib import pyplot as plt
import numpy as np


def save_checkpoint(model, checkpoint_name, out_folder):
    torch.save(model.state_dict(), os.path.join(out_folder, f'{checkpoint_name}.pth'))
    msg = f"{checkpoint_name}.pth saved\n\n"
    logger(msg, os.path.join(out_folder,'log.txt'))

def load_checkpoint(model, savefile, log_file, gpu, _log=False):
    saved_state = torch.load(savefile, map_location=torch.device("cpu"))
    model.load_state_dict(saved_state)
    if _log:
        logger(f"Model in {savefile} loaded\n", log_file)

    if gpu != -1:
        model = model.cuda()
    return model

def logger(msg, log_path):
    print(msg)
    open(log_path, 'a').write(msg)


def calculate_cost(model, name):
    export_onnx_path = f"./Models/{name}_export.onnx"
    final_onnx_path = f"./Models/{name}_final.onnx"
    cost_dict_path = f"./Models/{name}_cost.json"
    BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path, opset_version = 9)
    inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
                preprocess=True, discount_sparsity=True)

    with open(cost_dict_path, 'r') as f:
        inference_cost_dict = json.load(f)

    bops = int(inference_cost_dict["total_bops"])
    w_bits = int(inference_cost_dict["total_mem_w_bits"])

    bops_baseline = 807699904
    w_bits_baseline = 1244936
    
    bops_ratio = bops/bops_baseline
    w_bits_ratio = w_bits/w_bits_baseline
    score = 0.5 * bops_ratio + 0.5 * w_bits_ratio
    
    print(f"Ops is {0.5*bops_ratio}, w_bits is {0.5*w_bits_ratio}")
    print("Normalized inference cost score: %f" % score)
    return bops_ratio, w_bits_ratio, score

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def sparsity_report(model, layerwise_report=False):
    overall_sparsity = []
    name_sparsity_pair = []
    for name, param in model.named_parameters():
        numerator = float(torch.sum(param == 0))
        denumerator = float(param.nelement())
        sparsity = 100 * float(numerator / denumerator)
        if name.endswith('weight'):
            name_sparsity_pair.append((name, sparsity))
            overall_sparsity.append([numerator, denumerator])

    global_sparsity = np.sum(np.array(overall_sparsity[::2]), 0)
    print(f"Global sparsity is: {global_sparsity[0] / global_sparsity[1]}\n\n")
    print(f"Total number of parameters is {global_sparsity[1]}")
    print(f"Total number of parameters after pruning is {global_sparsity[1] - global_sparsity[0]}")
    if layerwise_report == True:
        outputs = name_sparsity_pair[::2]
        for output in outputs:
            print(f"Sparsity in {output[0]} is {output[1]:2f}%")