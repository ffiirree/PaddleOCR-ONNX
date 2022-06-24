import torchvision

def list_torch_models():
    model_list = torchvision.models.__dict__
    return [name for name in model_list
            if name.islower() and not name.startswith("__")
            and callable(model_list[name])]
            