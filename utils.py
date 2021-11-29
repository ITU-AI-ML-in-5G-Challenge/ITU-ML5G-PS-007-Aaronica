import torch
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from finn.util.inference_cost import inference_cost
import json
import os


def save_checkpoint(model, checkpoint_name, out_folder):
    torch.save(model.state_dict(), os.path.join(out_folder, f'{checkpoint_name}.pth'))
    msg = f"{checkpoint_name}.pth saved\n\n"
    logger(msg, os.path.join(out_folder,'log.txt'))




def load_checkpoint(model, savefile, log_file, gpu):
    saved_state = torch.load(savefile, map_location=torch.device("cpu"))
    model.load_state_dict(saved_state)
    logger(f"Model in {savefile} loaded\n", log_file)

    if gpu != -1:
        model = model.cuda()
    return model

def logger(msg, log_path):
    print(msg)
    open(log_path, 'a').write(msg)


def calculate_cost(model):
    export_onnx_path = "model_export.onnx"
    final_onnx_path = "model_final.onnx"
    cost_dict_path = "model_cost.json"
    BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path, opset_version = 9)
    inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
                preprocess=True, discount_sparsity=True)

    with open(cost_dict_path, 'r') as f:
        inference_cost_dict = json.load(f)

    bops = int(inference_cost_dict["total_bops"])
    w_bits = int(inference_cost_dict["total_mem_w_bits"])

    bops_baseline = 807699904
    w_bits_baseline = 1244936
    print(f"Ops is {0.5*(bops/bops_baseline)}, w_bits is {0.5*(w_bits/w_bits_baseline)}")
    score = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline)
    print("Normalized inference cost score: %f" % score)
    return score
