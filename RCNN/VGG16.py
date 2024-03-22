# from PIL import Image


# with Image.open("/Users/eishahemchand/Mineral-Intelligence/png docs/5594238_83222099_docimage_actual.png") as img:

#   defining the imports 


import matplotlib.pyplot as plt
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
dataset = CustomDataset(annotations_file='/Users/eishahemchand/Mineral-Intelligence/final_output_signature.json',
                        img_dir='/Users/eishahemchand/Mineral-Intelligence/png docs',
                        transform=transform)


class CustomVGG16(nn.Module):
    def __init__(self, classifier_layers):
        super(CustomVGG16, self).__init__()
        # Load the pre-trained VGG16 model
        original_vgg16 = models.vgg16(pretrained=True)
        # Remove the classifier
        self.features = original_vgg16.features
        # Freeze the layers
        for param in self.features.parameters():
            param.requires_grad = False

        # Initialize classifier from argument

        #self.classifier = nn.Sequential(*classifier_layers)

        
        # # Custom layers for bounding box regression
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512 * 7 * 7, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4),  # Output 4 coordinates
        #     nn.Sigmoid()  # Assuming coordinates are normalized between [0, 1]
        # )
    
    def forward(self, x):
        x = self.features(x)  # Pass through the feature extractor
        #x = self.classifier(x)  # Pass through the custom classifier
        return x




model = CustomVGG16([
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 128),
    nn.BatchNorm1d(128),  # BatchNorm layer
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),  # BatchNorm layer
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32,4),
    nn.Sigmoid()
])
print(model)







layer_options = [[
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 128),
    nn.LeakyReLU(0.01),
    nn.Linear(128, 64),
    nn.LeakyReLU(0.01),
    nn.Linear(64, 32),
    nn.LeakyReLU(0.01),
    nn.Linear(32, 4),
    nn.Sigmoid()
],[
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
    nn.Sigmoid()
],[
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 128),
    nn.ReLU(),
    nn.Dropout(0.5),  # Dropout layer
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32,4),
    nn.Sigmoid()
],[
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 128),
    nn.BatchNorm1d(128),  # BatchNorm layer
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),  # BatchNorm layer
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32,4),
    nn.Sigmoid()
] ]


# # Optimizer and Loss Function
# optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)  # Only train the custom layers
# loss_fn = nn.MSELoss()
optimizer_options = ['Adam', 'SGD']  # Extend as needed
loss_options = ['MSE', 'L1']  # Extend as needed



# Assuming 'dataset' is an instance of CustomDataset
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)



# optimizer = get_optimizer(model, optimizer_name='Adam', lr=1e-4)
# loss_fn = get_loss_function(name='MSE')

# OLD Training loop
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

def get_loss_function(name='MSE'):
    if name == 'MSE':
        return nn.MSELoss()
    elif name == 'L1':
        return nn.L1Loss()
    # Add more loss functions as needed
    else:
        raise ValueError("Unsupported loss function")





#create optimizer chooser function 

def get_optimizer(model, optimizer_name='Adam', lr=1e-4):
    if optimizer_name == 'Adam':
        return optim.Adam(model.classifier.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)
    # Add more optimizers as needed
    else:
        raise ValueError("Unsupported optimizer")



#run experiments 

def run_experiments(layer_options, optimizer_options, loss_options, epochs=25):
    results = {}
    for layer_config in layer_options:
        model = CustomVGG16(classifier_layers=layer_config)
        for optimizer_name in optimizer_options:
            optimizer = get_optimizer(model, optimizer_name)
            for loss_name in loss_options:
                loss_fn = get_loss_function(loss_name)
                loss_history = []
                
                model.train()
                for epoch in range(epochs):
                    epoch_loss = 0
                    for imgs, targets in train_loader:
                        optimizer.zero_grad()
                        outputs = model(imgs)
                        loss = loss_fn(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
                    avg_epoch_loss = epoch_loss / len(train_loader)
                    loss_history.append(avg_epoch_loss)
                
                # Key to identify the experiment
                key = f"{type(layer_config).__name__}_{optimizer_name}_{loss_name}"
                results[key] = loss_history
                print(f"Completed: {key}")
    return results




#plot results 


def plot_results(results):
    plt.figure(figsize=(10, 8))
    for key, loss_history in results.items():
        plt.plot(loss_history, label=key)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs for Various Configurations')
    plt.legend()
    plt.show()










# Train the model
results = run_experiments(layer_options, optimizer_options, loss_options, epochs=25)
plot_results(results)




                      













#define the custom model with non trainable layers ie - last layer 















