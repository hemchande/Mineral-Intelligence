import torchvision
import torch
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image, ImageDraw


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

model.eval()

x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

predictions = model(x)


# images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)

# print(boxes)


# boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]

# labels = torch.randint(1, 91, (4, 11))

# images = list(image for image in images)


# targets = []

# for i in range(len(images)):
#     d = {}
#     d['boxes'] = boxes[i]
#     d['labels'] = labels[i]
#     targets.append(d)

# output = model(images, targets)


# model.eval()

# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

# predictions = model(x)

print(predictions)



def draw_boxes(image_path, boxes):
    # Load image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Draw each box
    for box in boxes:
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red")

    img.show()

# Example usage
boxes = [(2000, 550, 1800, 1900), (1120, 870, 1180, 3140)]  # Replace with your model's output

draw_boxes("/Users/eishahemchand/Mineral-Intelligence/png docs/5594241_82095055_docimage_actual.png", boxes)
img = Image.open("/Users/eishahemchand/Mineral-Intelligence/png docs/5594241_82095055_docimage_actual.png")
print(img.size)

