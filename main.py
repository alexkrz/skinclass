import torch

import flash
from flash.image import ImageClassificationData, ImageClassificationInputTransform, ImageClassifier

# 1. Create the DataModule
datamodule = ImageClassificationData.from_folders(
    train_folder="/home/kti03/Data/ISIC2018/train",
    val_folder="/home/kti03/Data/ISIC2018/val",
    batch_size=128,
    num_workers=16,
    transform=ImageClassificationInputTransform(image_size=(224, 224)),
)

# 2. Build the task
model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=10, accelerator="gpu", devices=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict what's on a few images! ants or bees?
datamodule = ImageClassificationData.from_files(
    predict_files=[
        "/home/kti03/Data/ISIC2018/test_images/ISIC_0034524.jpg",
        "/home/kti03/Data/ISIC2018/test_images/ISIC_0034525.jpg",
        "/home/kti03/Data/ISIC2018/test_images/ISIC_0034526.jpg",
    ],
    batch_size=3,
)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")
