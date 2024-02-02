import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2



def calculate_initial_bounding_boxes(data_points, labels, K):
    bounding_boxes = []
    for i in range(K):
        cluster_points = data_points[labels == i]
        min_x, min_y = np.min(cluster_points, axis=0)
        max_x, max_y = np.max(cluster_points, axis=0)
        bounding_boxes.append((min_x, min_y, max_x, max_y))
    return bounding_boxes

def check_overlap(box1, box2):
    # Check if box1 and box2 overlap
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def adjust_boxes(bounding_boxes):
    for i in range(len(bounding_boxes)):
        for j in range(i+1, len(bounding_boxes)):
            if check_overlap(bounding_boxes[i], bounding_boxes[j]):
                # Simple adjustment: shrink the boxes
                box1 = list(bounding_boxes[i])
                box2 = list(bounding_boxes[j])

                # Reduce size by 10% for overlapping boxes
                reduction = 0.1
                box1_width, box1_height = box1[2] - box1[0], box1[3] - box1[1]
                box2_width, box2_height = box2[2] - box2[0], box2[3] - box2[1]

                box1[0] += box1_width * reduction
                box1[1] += box1_height * reduction
                box1[2] -= box1_width * reduction
                box1[3] -= box1_height * reduction

                box2[0] += box2_width * reduction
                box2[1] += box2_height * reduction
                box2[2] -= box2_width * reduction
                box2[3] -= box2_height * reduction

                bounding_boxes[i] = tuple(box1)
                bounding_boxes[j] = tuple(box2)

def optimize_boxes(bounding_boxes, centroids):
    # Placeholder for optimization logic
    pass





# Assuming the image is 1200x1200 pixels
image_size = 1200

# Randomly generate data points
# For demonstration purposes, let's create 500 data points
num_points = 500
data_points = np.random.rand(num_points, 2) * image_size

# Number of clusters
K = 3

# Apply KMeans clustering
kmeans = KMeans(n_clusters=K)
kmeans.fit(data_points)

# Cluster centers
centroids = kmeans.cluster_centers_

# Assignments of data points to clusters
labels = kmeans.labels_


# # Find extremes for each cluster to define bounding boxes
# bounding_boxes = []
# for i in range(K):
#     cluster_points = data_points[labels == i]
#     min_x, min_y = np.min(cluster_points, axis=0)
#     max_x, max_y = np.max(cluster_points, axis=0)
#     bounding_boxes.append([(min_x, min_y), (max_x, max_y)])

# # Plotting with bounding boxes
# plt.figure(figsize=(8, 8))
# for i, box in enumerate(bounding_boxes):
#     plt.scatter(data_points[labels == i, 0], data_points[labels == i, 1], cmap='rainbow')
#     plt.plot([box[0][0], box[1][0], box[1][0], box[0][0]])

# plt.show()


# # Plotting
# plt.figure(figsize=(8, 8))
# plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='rainbow')
# plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x')
# plt.xlim(0, image_size)
# plt.ylim(0, image_size)
# plt.title("K-means Clustering on 1200x1200 Pixel Data")
# plt.show()

# Example usage
# Assuming data_points, labels, and K are defined as in your previous example
bounding_boxes = calculate_initial_bounding_boxes(data_points, labels, K)

while any(check_overlap(bounding_boxes[i], bounding_boxes[j]) for i in range(K) for j in range(i+1, K)):
    adjust_boxes(bounding_boxes)

# Placeholder for optimization (not implemented)
optimize_boxes(bounding_boxes, centroids)

def get_box_coordinates(bounding_boxes):
    """
    Extract the coordinates from the bounding boxes.

    Parameters:
    bounding_boxes (list of tuples): List of bounding boxes, each represented as a tuple (min_x, min_y, max_x, max_y).

    Returns:
    list of tuples: List of coordinates for each bounding box.
    """
    coordinates = []
    for box in bounding_boxes:
        min_x, min_y, max_x, max_y = box
        # Coordinates format: (top_left, top_right, bottom_right, bottom_left)
        top_left = (min_x, min_y)
        top_right = (max_x, min_y)
        bottom_right = (max_x, max_y)
        bottom_left = (min_x, max_y)
        coordinates.append((top_left, top_right, bottom_right, bottom_left))
    return coordinates

# Example usage with the bounding boxes calculated earlier
box_coordinates_final = get_box_coordinates(bounding_boxes)
box_coordinates_final







# Continuing from the previous KMeans clustering code

# Initialize list to store bounding box coordinates
# bounding_boxes = []

# for i in range(K):
#     cluster_points = data_points[labels == i]
    
#     # Find the min and max points for each cluster
#     min_x, min_y = np.min(cluster_points, axis=0)
#     max_x, max_y = np.max(cluster_points, axis=0)
    
#     # Define the bounding box [upper-left-x, upper-left-y, lower-right-x, lower-right-y]
#     bounding_box = [min_x, min_y, max_x, max_y]
#     bounding_boxes.append(bounding_box)

# Plotting with bounding boxes
plt.figure(figsize=(8, 8))
plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x')

# Draw the bounding boxes
for bbox in bounding_boxes:
    plt.plot([bbox[0], bbox[0], bbox[2], bbox[2], bbox[0]], 
             [bbox[1], bbox[3], bbox[3], bbox[1], bbox[1]], 'k-', linewidth=2)

plt.xlim(0, image_size)
plt.ylim(0, image_size)
plt.title("K-means Clustering with Bounding Boxes on 1200x1200 Pixel Data")
plt.show()


# Load the image
image_path = '/Users/eishahemchand/Mineral-Intelligence/png docs/2c91ef63-6e85-4225-ac36-20f0416647c0.png'  # Replace with your image path
image = cv2.imread(image_path)

# Draw each bounding box
for box in box_coordinates_final:
    top_left, top_right, bottom_right, bottom_left = box
    # OpenCV rectangle function requires top-left and bottom-right coordinates
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green box with 2px thickness

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

# Optionally, save the image
output_path = 'output_image_with_boxes.png'  # Replace with your desired output path
cv2.imwrite(output_path, image)

