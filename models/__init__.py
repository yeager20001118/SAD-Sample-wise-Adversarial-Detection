from .semantic_resnet import *

def load_model(model_arch, model_path):
    if model_arch == "Res18":
        model = semantic_ResNet18()
        model.load_state_dict(torch.load(model_path))
        model.eval()

    return model