from typing import Callable, Tuple, Union

import torch
from torchvision import transforms as T

import flash
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.io.input_transform import InputTransform
from dataclasses import dataclass


@dataclass
class ISICInputTransform(InputTransform):
    org_img_size: Tuple[int, int] = (600, 450)
    input_img_size: Tuple[int, int] = (224, 224)
    mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)

    def per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose(
                        [
                            T.ToTensor(),
                            T.RandomCrop(min(self.org_img_size)),
                            T.Resize(self.input_img_size),
                            T.Normalize(self.mean, self.std),
                        ],
                    ),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )

    def train_per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose(
                        [
                            T.ToTensor(),
                            T.RandomCrop(min(self.org_img_size)),
                            T.Resize(self.input_img_size),
                            T.Normalize(self.mean, self.std),
                            T.RandomHorizontalFlip(),
                            T.ColorJitter(),
                            T.RandomAutocontrast(),
                            T.RandomPerspective(),
                        ]
                    ),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )
