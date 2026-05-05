import os
import json
import torch    
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from collections import defaultdict

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weight_into_gpt(gpt, params):
    gpt.pos_emb_weight = assign(gpt.pos_emb_weight, params["wpe"])
    gpt.tok_emb_weight = assign(gpt.token_emb_weight, params["wte"]) 

    for b in range(len(params["blocks"])):
        q_w,k_w,v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.weight = assign(
            gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_key.weight = assign(
            gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_value.weight = assign(
            gpt.trf_blocks[b].attn.W_value.weight, v_w.T)
        
        q_b,k_b,v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.bias = assign(
            gpt.trf_blocks[b].attn.W_query.bias, q_b)
        gpt.trf_blocks[b].attn.W_key.bias = assign(
            gpt.trf_blocks[b].attn.W_key.bias, k_b)
        gpt.trf_blocks[b].attn.W_value.bias = assign(
            gpt.trf_blocks[b].attn.W_value.bias, v_b)
        
        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out.weight, (
            params["blocks"][b]["attn"]["c_proj"])["w"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias, (
            params["blocks"][b]["attn"]["c_proj"])["b"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, (
            params["blocks"][b]["mlp"]["c_fc"])["w"].T) 
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, (
            params["blocks"][b]["mlp"]["c_fc"])["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, (
            params["blocks"][b]["mlp"]["c_proj"])["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, (  
            params["blocks"][b]["mlp"]["c_proj"])["b"])
        
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, (
            params["blocks"][b]["ln_1"]["g"]))
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, (
            params["blocks"][b]["ln_1"]["b"]))
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, (
            params["blocks"][b]["ln_2"]["g"]))
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, (
            params["blocks"][b]["ln_2"]["b"]))
        
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def load_gpt2_params_from_tf_ckpt(ckpt_path,settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
        
    for name,_ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, mane))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    total_params = count_params(params)
   
    print(f"Total parameters in file: {total_params}")
   
    return params

def count_params(d):
    total = 0
    for v in d.values():
        if isinstance(v, dict):
            total += count_params(v)
        elif isinstance(v, list):
            for item in v:
              total += count_params(item)
        else:
            total += v.size
    return total


def get_settings_and_params(model_dir):
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def debug_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    param_counts = defaultdict(int)

    for name, param in model.named_parameters():
        top_level = name.split('.')[0]
        param_counts[top_level] += param.numel()

    for k, v in param_counts.items():
        print(f"{k}: {v} parameters")    
           