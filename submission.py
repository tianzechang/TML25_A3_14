import requests
import torch
import torch.nn as nn
import os
from torchvision import models

#### Tests ####
# (these are the assertions being ran on the eval endpoint for every submission)

allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}
with open("robust_model.pt", "rb") as f:
    try:
        model: torch.nn.Module = allowed_models["resnet34"](weights=None)
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)
    except Exception as e:
        raise Exception(
            f"Invalid model class, {e=}, only {allowed_models.keys()} are allowed",
        )
    try:
        state_dict = torch.load(f, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        out = model(torch.randn(1, 3, 32, 32))
    except Exception as e:
        raise Exception(f"Invalid model, {e=}")

    assert out.shape == (1, 10), "Invalid output shape"

input()
# Send the model to the server, replace the string "TOKEN" with the string of token provided to you
response = requests.post("http://34.122.51.94:9090/robustness", files={"file": open("robust_model.pt", "rb")}, headers={"token": "34811541", "model-name": "resnet34"})

# Should be 400, the clean accuracy is too low
print(response.json())