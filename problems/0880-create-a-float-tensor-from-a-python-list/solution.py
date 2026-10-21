import torch

def to_float_tensor(values):
    v_t=torch.tensor(values)
    return v_t.float()
