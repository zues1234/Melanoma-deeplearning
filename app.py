from flask import Flask
from flask import render_template
from flask import request

import os
import pretrainedmodels
import torch.nn as nn
from tqdm import tqdm

from PIL import Image
from PIL import ImageFile

import numpy as np
import cv2

import torch

import torch.nn.functional as F 
import albumentations as albumentations





app = Flask(__name__)
upload_folder = 'F:\Data Science\workspace\static'
DEVICE = "cpu"

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = torch.sigmoid(self.l0(x))
        loss = 0
        return out, loss

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize, augmentations=None, backend="pil", channel_first=True,):
        super(ClassificationDataset, self).__init__()
        
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if os.path.isfile(self.image_paths[item]):
            if self.backend == "pil":
                image = Image.open(self.image_paths[item])
                if self.resize is not None:
                    image = image.resize(
                        (self.resize[1], self.resize[0]), resample=Image.BILINEAR
                    )
                image = np.array(image)
                if self.augmentations is not None:
                    augmented = self.augmentations(image=image)
                    image = augmented["image"]
            elif self.backend == "cv2":
                image = cv2.imread(self.image_paths[item])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.resize is not None:
                    image = cv2.resize(
                        image,
                        (self.resize[1], self.resize[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                if self.augmentations is not None:
                    augmented = self.augmentations(image=image)
                image = augmented["image"]
            else:
                raise Exception("Backend not implemented")
            if self.channel_first:
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets),
        }

def predict_fn(model, data_loader, device):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for data in tk0:
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, _ = model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions

def predict(image_path, model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    predictions = predict_fn(
        model,
        test_loader,
        DEVICE
    )
    return np.vstack((predictions)).ravel()

        
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, MODEL)[0]
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)

if __name__ == "__main__":
    MODEL = SEResNext50_32x4d(pretrained=None)
    MODEL.load_state_dict(torch.load("model_fold_4.bin", map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    app.run(port=22000, debug=True)