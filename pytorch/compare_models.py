import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import flash
from flash.image import ImageClassificationData, ImageClassifier

import PIL
import numpy as np
import matplotlib.pyplot as plt

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
traced_model = torch_model.to_torchscript(method="trace", example_inputs=example_input)
out = traced_model(example_input)
print(out.reshape(-1))

# Load the test image and resize to 224, 224.
img_path = Path(os.environ["ISIC_DATA_PATH"]) / "test_mel_nev" / "melanoma" / "ISIC_0034529.jpg"
org_img = PIL.Image.open(img_path)
pil_transforms = T.Compose(
    [
        T.CenterCrop(min(org_img.size)),
        T.Resize([224, 224]),
    ],
)
torch_transforms = T.Compose(
    [
        T.CenterCrop(min(org_img.size)),
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ],
)
img = pil_transforms(org_img)
# plt.imshow(img)
# plt.show()

# Invoke prediction and print outputs.
torch_out = traced_model(torch_transforms(org_img).unsqueeze(0))

torch_out_np = torch_out.detach().numpy().squeeze()
top_indices = np.argsort(-torch_out_np)[:2]
print("torch top predictions: ")
for i in range(2):
    idx = top_indices[i]
    score_value = torch_out_np[idx]
    class_id = datamodule.labels[idx]
    print("class name: {}, raw score value: {}".format(class_id, score_value))


# Load Core ML Model
mlmodel = ct.models.MLModel("isic_resnet18_2cl.mlmodel")

# Get the protobuf spec of the model.
spec = mlmodel.get_spec()
for out in spec.description.output:
    if out.type.WhichOneof("Type") == "dictionaryType":
        coreml_dict_name = out.name
        break

# Make a prediction with the Core ML version of the model. Only supported on MacOS!
coreml_out_dict = mlmodel.predict({"input_1": img})
print("coreml predictions: ")
print("top class label: ", coreml_out_dict["classLabel"])

coreml_prob_dict = coreml_out_dict[coreml_dict_name]

values_vector = np.array(list(coreml_prob_dict.values()))
keys_vector = list(coreml_prob_dict.keys())
top_indices_coreml = np.argsort(-values_vector)[:2]
for i in range(2):
    idx = top_indices_coreml[i]
    score_value = values_vector[idx]
    class_id = keys_vector[idx]
    print("class name: {}, raw score value: {}".format(class_id, score_value))
