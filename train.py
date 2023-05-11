import torch

import flash
from flash.image import ImageClassificationData, ImageClassifier

# Custom imports
from transforms import ISICInputTransform

# 1. Create the DataModule
datamodule = ImageClassificationData.from_folders(
    train_folder="/home/kti03/Data/ISIC2018/train",
    val_folder="/home/kti03/Data/ISIC2018/val",
    batch_size=64,
    num_workers=12,
    transform=ISICInputTransform(),
)

# 2. Build the task
model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=10, accelerator="gpu", devices=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict what's on a few images! ants or bees?
datamodule = ImageClassificationData.from_files(
    predict_files=[
        "/home/kti03/Data/ISIC2018/test/MEL/ISIC_0034529.jpg",
        "/home/kti03/Data/ISIC2018/test/MEL/ISIC_0034548.jpg",
        "/home/kti03/Data/ISIC2018/test/MEL/ISIC_0034572.jpg",
    ],
    batch_size=3,
    num_workers=12,
    transform=ISICInputTransform(),
)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("isic_resnet18.pt")
