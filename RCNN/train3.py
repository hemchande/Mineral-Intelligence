import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data.dataloader import default_collate
import json
import os
import torch.nn as nn
import torch.optim as optim
from PIL import Image 
# from labelbox import Client
# from labelbox.schema.conflict_resolution_strategy import ConflictResolutionStrategy
# from labelbox.schema.identifiables import DataRowIds

print(torch.__version__)
print(torchvision.__version__)
#define a custom model

class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        # Load the pre-trained VGG16 model
        original_vgg16 = fasterrcnn_resnet50_fpn(pretrained=True)
        #original_vgg16 = models.vgg16(pretrained=True)
        # Remove the classifier
        self.features = original_vgg16.features
        # Freeze the layers
    

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


def collate_fn(batch):
    images, targets = list(zip(*batch))  # Unzip the batch
    images = default_collate(images)  # Use the default collate function for images
    # No need to collate targets as we want them in a list of dictionaries
    return images, list(targets)








##FasterRCNN model using Smooth L1loss to not be as disrupted by outliers 

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_label['filename'])
        image = Image.open(img_path).convert("RGB")

        # Assuming 'coords' contains bounding box coordinates [startx, starty, endx, endy]

         # Extract the bounding box coordinates
        startx = img_label['startx']
        starty = img_label['starty']
        endx = img_label['endx']
        endy = img_label['endy']
        if (startx == 0 or startx == 0.0 or abs(endx - startx) <= 0.0 or starty == 0 or endy == 0.0 or abs(endy - starty) <= 0.0 ):
            boxes = torch.tensor([1.0, 1.0, 1.1, 1.1], dtype=torch.float).unsqueeze(0)
        #coordinates = [startx, starty, endx, endy]  # Create a list of coordinates
        else:
            boxes = torch.tensor([startx, starty, endx, endy], dtype=torch.float).unsqueeze(0)
    
        # boxes = torch.tensor([startx, starty, endx, endy], dtype=torch.float).unsqueeze(0)  # Add an extra dimension to match [N, 4]

    # For the labels, assuming there's only one class of object, you would assign the same label for each bounding box.
    # Here we use 1 since 0 is typically reserved for the background class in object detection datasets.
        labels = torch.ones((1,), dtype=torch.int64)  # Assuming a single class, you might adjust this based on your actual dataset



        #coordinates = img_label['coords']

        if self.transform:
            image = self.transform(image)

         # Convert coordinates to a tensor
        #coordinates = torch.tensor(coordinates, dtype=torch.float)

        target = {'boxes': boxes, 'labels': labels}


        return image, target

# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load your custom dataset
# dataset = CustomDataset(annotations_file='/Users/eishahemchand/Mineral-Intelligence/final_output_signature.json',
#                         img_dir='/Users/eishahemchand/Mineral-Intelligence/png docs',
#                         transform=transform)

# # DataLoader for batch processing
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Load the pre-trained Faster R-CNN model
# model = fasterrcnn_resnet50_fpn(pretrained=True)


# in_features_cls = model.roi_heads.box_predictor.cls_score.in_features

# # Number of classes is 2 (background + your class)
# num_classes = 2

# # Create a new FastRCNNPredictor with the desired number of classes
# # For the classification head
# model.roi_heads.box_predictor.cls_score = nn.Linear(in_features_cls, num_classes)

# # For the bounding box head, if deciding to adjust for a single class
# # This step is optional and typically not done, but shown here for completeness
# in_features_bbox = model.roi_heads.box_predictor.bbox_pred.in_features
# # Traditionally, this remains designed for multiple classes, but for one class + background, theoretically, it would be 4
# # However, to align with the typical setup, you might keep it as is or adjust depending on your specific requirements
# model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features_bbox, num_classes * 4)  # 4 coordinates for each class


#defining helper function to define custom layer freezing/properties
#types = ["train_all_layers", "train_box_heads_predictor","train_roi_heads","alter_roi_heads","alter_box_heads_predictor"]

def define_architecture(model,type):
    if type == "train_all_layers":
        for param in model.parameters():
            param.requires_grad = True

        return model
    if type == "train_box_heads_predictor":
        in_features_cls = model.roi_heads.box_predictor.cls_score.in_features


        num_classes = 1

# Create a new FastRCNNPredictor with the desired number of classes
# For the classification head
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features_cls, num_classes)

# For the bounding box head, if deciding to adjust for a single class
# This step is optional and typically not done, but shown here for completeness
        in_features_bbox = model.roi_heads.box_predictor.bbox_pred.in_features
# Traditionally, this remains designed for multiple classes, but for one class + background, theoretically, it would be 4
# However, to align with the typical setup, you might keep it as is or adjust depending on your specific requirements
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features_bbox, num_classes * 4)  # 4 coordinates for each class

        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True

        return model
    
    if type == "train_roi_heads":

        in_features_f6 = model.roi_heads.box_head.fc6
        in_features_f7 = model.roi_heads.box_head.fc7

        model.roi_heads.box_head.fc6 = nn.Linear(in_features_f6,16)

        model.roi_heads.box_head.fc7 = nn.Linear(in_features_f7,in_features_f7)


        in_features_cls = model.roi_heads.box_predictor.cls_score.in_features


        num_classes = 1

# Create a new FastRCNNPredictor with the desired number of classes
# For the classification head
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features_cls, num_classes)

# For the bounding box head, if deciding to adjust for a single class
# This step is optional and typically not done, but shown here for completeness
        in_features_bbox = model.roi_heads.box_predictor.bbox_pred.in_features
# Traditionally, this remains designed for multiple classes, but for one class + background, theoretically, it would be 4
# However, to align with the typical setup, you might keep it as is or adjust depending on your specific requirements
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features_bbox, num_classes * 4)  # 4 coordinates for each class

        for param in model.parameters():
            param.requires_grad() == False

        for param in model.roi_heads.box_head.parameters():
            param.requires_grad() == True

        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad() == True

        print(model.parameters())

        return model
        #print(model.parameters())
        

    if type == "alter_roi_heads":
        in_features_f6 = model.roi_heads.box_head.fc6
        in_features_f7 = model.roi_heads.box_head.fc7

        model.roi_heads.box_head.fc6 = nn.Linear(in_features_f6,16)

        model.roi_heads.box_head.fc7 = nn.Linear(in_features_f7,in_features_f7)


        in_features_cls = model.roi_heads.box_predictor.cls_score.in_features


        num_classes = 1

# Create a new FastRCNNPredictor with the desired number of classes
# For the classification head
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features_cls, num_classes)

# For the bounding box head, if deciding to adjust for a single class
# This step is optional and typically not done, but shown here for completeness
        in_features_bbox = model.roi_heads.box_predictor.bbox_pred.in_features
# Traditionally, this remains designed for multiple classes, but for one class + background, theoretically, it would be 4
# However, to align with the typical setup, you might keep it as is or adjust depending on your specific requirements
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features_bbox, num_classes * 4)  # 4 coordinates for each class

        for param in model.parameters():
            param.requires_grad = True


        print(model.parameters())

        return model



    if type == "alter_box_heads_predictor":
        in_features_cls = model.roi_heads.box_predictor.cls_score.in_features


        num_classes = 1

# Create a new FastRCNNPredictor with the desired number of classes
# For the classification head
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features_cls, num_classes)

# For the bounding box head, if deciding to adjust for a single class
# This step is optional and typically not done, but shown here for completeness
        in_features_bbox = model.roi_heads.box_predictor.bbox_pred.in_features
# Traditionally, this remains designed for multiple classes, but for one class + background, theoretically, it would be 4
# However, to align with the typical setup, you might keep it as is or adjust depending on your specific requirements
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features_bbox, num_classes * 4)  # 4 coordinates for each class

        for param in model.parameters():
            param.requires_grad = True


        print(model.parameters())

        return model


    

# for param in model.parameters():
#     param.requires_grad = False


# # Unfreeze the box predictor layers (classification and bounding box regression)
# for param in model.roi_heads.box_predictor.parameters():
#     param.requires_grad = True

# # Get the number of input features to the final classification layer
# in_features = model.roi_heads.box_predictor.cls_score.in_features


# # Replace the pre-trained head with a new one (num_classes is 2 for your object + background)
# num_classes = 2  # 1 class (your object) + background
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# in_features_cls = model.roi_heads.box_predictor.cls_score.in_features

# # Number of classes is 2 (background + your class)
# num_classes = 2

# # Create a new FastRCNNPredictor with the desired number of classes
# # For the classification head
# model.roi_heads.box_predictor.cls_score = nn.Linear(in_features_cls, num_classes)

# # For the bounding box head, if deciding to adjust for a single class
# # This step is optional and typically not done, but shown here for completeness
# in_features_bbox = model.roi_heads.box_predictor.bbox_pred.in_features
# # Traditionally, this remains designed for multiple classes, but for one class + background, theoretically, it would be 4
# # However, to align with the typical setup, you might keep it as is or adjust depending on your specific requirements
# model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features_bbox, num_classes * 4)  # 4 coordinates for each class


# model = fasterrcnn_resnet50_fpn(pretrained=True)


# Define loss function and optimizer
#criterion = ...  # Define your loss function
# criterion = nn.SmoothL1Loss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Train the model (you may need to adapt this part based on your specific dataset and requirements)
# model.train()
# for epoch in range(20):
#     for images, targets in train_loader:
#         optimizer.zero_grad()
#         result = model(images)
#         #loss = criterion(result,targets)
#         loss_dict = model(images, targets)
#         #losses = sum(loss for loss in loss_dict.values())
#         losses = sum(loss for loss in loss_dict.values())

#         optimizer.zero_grad()

#         #print(loss + " " + "epoch" + epoch)
#         #print(result)

        #print(t)
        #loss.backward()
        #optimizer.step()

# print(predictions)


#defining a train function 

def trainModel(epochs,model, arch,train_loader,filename):



    model = define_architecture(model,arch)

    model = normalizeModel(model)


    model.train()


    #defining parameters 
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(epochs):
        for images, targets in train_loader:
            assert isinstance(targets, list) and all(isinstance(t, dict) for t in targets), "Targets must be a list of dictionaries."
            # Proceed with training...
            optimizer.zero_grad()
            print(targets)

            #result = model(images)
        #loss = criterion(result,targets)
            loss_dict = model(images, targets)
        #losses = sum(loss for loss in loss_dict.values())
            losses = sum(loss for loss in loss_dict.values())
            print( losses)
            print("new epoch")

            losses.backward()



            #loss_dict.backward()
            
            optimizer.step()

    
    



        #print(loss + " " + "epoch" + epoch)
        #print(result)
            print(targets)


    #model.save("")

    model.eval()

    torch.save(model.state_dict(), '{architecture}-{epochs}-{filename}model_state_dict.pth')



    


#normalize the bounding box/predictor layers:

def normalizeModel(model):
       


    model.roi_heads.box_head.bn6 = nn.BatchNorm1d(16)

    print(model.roi_heads.box_head.fc7)

    in_features_bbox = model.roi_heads.box_head.fc7.in_features

    model.roi_heads.box_head.bn6 = nn.BatchNorm1d(in_features_bbox)

    return model




model = fasterrcnn_resnet50_fpn(pretrained=True)

#trainModel(20,model,"train_all_layers")


def initiate_training_loop():
   layer_config_types = ["train_all_layers", "train_box_heads_predictor","train_roi_heads","alter_roi_heads","alter_box_heads_predictor"]
   # Specify your directory path here
   file_directory = "/Users/eishahemchand/Mineral-Intelligence/RCNN/files"
   #img_directory = "/Users/eishahemchand/Mineral-Intelligence/RCNN/files/imgs"
   # Iterate through each file in the directory
   # Load your custom dataset
#    dataset = CustomDataset(annotations_file='/Users/eishahemchand/Mineral-Intelligence/final_output_signature.json',
#                         img_dir='/Users/eishahemchand/Mineral-Intelligence/png docs',
#                         transform=transform)

# # DataLoader for batch processing
#    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
   for filename in os.listdir(file_directory): 

    # Construct the full file path
        file_path = os.path.join(file_directory, filename)
        dataset = CustomDataset(annotations_file=file_path,
                        img_dir='/Users/eishahemchand/Mineral-Intelligence/png docs',
                        transform=transform)

# DataLoader for batch processing
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        for type1 in layer_config_types:
            trainModel(20,model,type1,train_loader=train_loader,filename=filename)

initiate_training_loop()





        









    






        
    






    


        













