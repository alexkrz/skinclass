# Conversion of Pytorch Model to Core ML following
# https://coremltools.readme.io/docs/convert-a-torchvision-model-from-pytorch

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

import flash
from flash.image import ImageClassificationData, ImageClassifier

# Custom imports
from transforms import ISICInputTransform

# Core ML
import coremltools as ct


# 1. Create the DataModule
datamodule = ImageClassificationData.from_folders(
    train_folder=Path(os.environ["ISIC_DATA_PATH"]) / "train_mel_nev",
    val_folder=Path(os.environ["ISIC_DATA_PATH"]) / "val_mel_nev",
    batch_size=64,
    num_workers=12,
    transform=ISICInputTransform(),
)

# 2. Build the task
lit_module = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

torch_model = lit_module.load_from_checkpoint("isic_resnet18_2cl.pt")
# Add softmax layer to return normalized confidence values
torch_model.adapter.head = nn.Sequential(torch_model.adapter.head, nn.Softmax(dim=1))
torch_model.eval()

# Trace the model with random data.
example_input = torch.rand(1, 3, 224, 224)
# traced_model = torch.jit.trace(torch_model, example_input)
# Fix "module not attached to trainer" bug according to https://github.com/Lightning-AI/lightning/issues/14036
traced_model = torch_model.to_torchscript(method="trace", example_inputs=example_input)
out = traced_model(example_input)
assert len(out.reshape(-1)) == len(datamodule.labels)

# Preprocess input for coreml
scale = 1 / (0.226 * 255.0)
bias = [-0.485 / (0.229), -0.456 / (0.224), -0.406 / (0.225)]

image_input = ct.ImageType(name="input_1", shape=example_input.shape, scale=scale, bias=bias)

# Using image_input in the inputs parameter:
# Convert to Core ML using the Unified Conversion API.
model = ct.convert(
    traced_model,
    inputs=[image_input],
    classifier_config=ct.ClassifierConfig(datamodule.labels),
    # compute_units=ct.ComputeUnit.CPU_ONLY,
)

# Save the converted model.
model.save("isic_resnet18_2cl.mlmodel")
# Print a confirmation message.
print("model converted and saved")
