import torch
from models.experimental import attempt_load


def load_model(args):
    device = 'cuda' if args.use_cuda else 'cpu'
    half = device != 'cpu'
    model = attempt_load(args.model_path, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    if device != 'cpu':
        model(torch.zeros(1, 3, args.imgsz, args.imgsz).to(device).type_as(next(model.parameters())))

    return model
