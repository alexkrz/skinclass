# Conversion of Pytorch Model to Core ML following
# https://coremltools.readme.io/docs/convert-a-torchvision-model-from-pytorch

import torch
import torchvision

import flash
from flash.image import ImageClassificationData, ImageClassifier

# Custom imports
from transforms import ISICInputTransform

# Core ML
import coremltools as ct

# 1. Create the DataModule
datamodule = ImageClassificationData.from_folders(
    train_folder="/home/kti03/Data/ISIC2018/train",
    val_folder="/home/kti03/Data/ISIC2018/val",
    batch_size=64,
    num_workers=12,
    transform=ISICInputTransform(),
)

# 2. Build the task
lit_module = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

torch_model = lit_module.load_from_checkpoint("isic_resnet18.pt").backbone
torch_model.eval()

# Trace the model with random data.
example_input = torch.rand(1, 3, 640, 480)
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)

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
model.save("isic_resnet18.mlmodel")
# Print a confirmation message.
print("model converted and saved")
