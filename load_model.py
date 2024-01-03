import torch

model_path = "gramformer.pt"
model = torch.load(model_path)

model.eval()