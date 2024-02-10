# from PIL import Image


# with Image.open("/Users/eishahemchand/Mineral-Intelligence/png docs/5594238_83222099_docimage_actual.png") as img:

#   defining the imports 



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import cv2
import os


#define the custom dataset 

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels[idx]['filename'])
        # image = Image.open(img_path).convert("RGB")
        # coordinates = self.img_labels[idx]['coords']  # Assuming coords is a list [x, y, width, height]
        # if self.transform:
        #     image = self.transform(image)
        # return image, torch.tensor(coordinates)
        img_label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_label['filename'])
        image = Image.open(img_path).convert("RGB")

        # Extract the bounding box coordinates
        startx = img_label['startx']
        starty = img_label['starty']
        endx = img_label['endx']
        endy = img_label['endy']
        coordinates = [startx, starty, endx, endy]  # Create a list of coordinates

        if self.transform:
            image = self.transform(image)

        # Convert coordinates to a tensor
        coordinates = torch.tensor(coordinates, dtype=torch.float)

        return (image,coordinates)


# Transformation: Resize and Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Assuming 'final_output.json' is structured properly for PyTorch
# with each entry having 'filename' and 'coords'
dataset = CustomDataset(annotations_file='/Users/eishahemchand/Mineral-Intelligence/final_output.json',
                        img_dir='/Users/eishahemchand/Mineral-Intelligence/png docs',
                        transform=transform)


class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        # Load the pre-trained VGG16 model
        original_vgg16 = models.vgg16(pretrained=True)
        # Remove the classifier
        self.features = original_vgg16.features
        # Freeze the layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Custom layers for bounding box regression
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # Output 4 coordinates
            nn.Sigmoid()  # Assuming coordinates are normalized between [0, 1]
        )
    
    def forward(self, x):
        x = self.features(x)  # Pass through the feature extractor
        x = self.classifier(x)  # Pass through the custom classifier
        return x

model = CustomVGG16()
print(model)

# Optimizer and Loss Function
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)  # Only train the custom layers
loss_fn = nn.MSELoss()



# Assuming 'dataset' is an instance of CustomDataset
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
def train(model, optimizer, loss_fn, epochs=25):
    model.train()
    for epoch in range(epochs):
        for imgs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")



# create loss chooser function 





#create optimizer chooser function 






# create epoch chooser function 








# Train the model
train(model, optimizer, loss_fn)




                      













#define the custom model with non trainable layers ie - last layer 















