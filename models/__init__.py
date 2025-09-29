from .resnet import *
import torch

def load_model(model_arch, model_path, semantic=True):
    if "Res18" in model_arch:
        model = RN18_10(semantic=semantic)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    return model