import labelbox
import json 

# Enter your Labelbox API key here
# LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbG01cjYyaTQwNGc0MDd5cWZ4ZjAxZnFiIiwib3JnYW5pemF0aW9uSWQiOiJjbG01cjYyaHEwNGczMDd5cWRmNzJoOGx5IiwiYXBpS2V5SWQiOiJjbHMzbTI3ZnkwYTE2MDd6eTM2dXgxcnFzIiwic2VjcmV0IjoiMGE1YTM5ZjY4ZjBjYzQ5ODMxYmM2NDlkZDYwNjE3M2MiLCJpYXQiOjE3MDY4MTU4MDQsImV4cCI6MjMzNzk2NzgwNH0.NI_GwF1KIszj6xKjU5CE2UvLi5EhJj2hmEXdi2hv3UA"

# PROJECT_ID = 'clrz7n3yt0dby072kffy1d4i1'
# client = labelbox.Client(api_key = LB_API_KEY)
# project = client.get_project(PROJECT_ID)
# labels = project.export_v2(params={
# 	"data_row_details": True,
# 	"metadata_fields": True,
# 	"attachments": True,
# 	"project_details": True,
# 	"performance_details": True,
# 	"label_details": True,
# 	"interpolated_frames": True
#   })


# # print(labels)


# import json

# # Initialize an empty list to hold the JSON objects
# json_list = []

# # Open the NDJSON file and read lines
# with open('/Users/eishahemchand/Mineral-Intelligence/RCNN/dataset/images/export-result.ndjson', 'r') as file:
#     for line in file:
#         # Parse each line as a JSON object and append to the list
#         json_list.append(json.loads(line))

# # Convert the list of JSON objects into a single JSON array
# json_array = json.dumps(json_list, indent=4)

# # Optionally, write the JSON array to a new JSON file
# with open('output.json', 'w') as json_file:
#     json_file.write(json_array)

# Print the result or use it as needed
# print(json_array)

# Function to extract bounding box for a specific label
def extract_bounding_box(json_data, label_name):
    bounding_boxes = []
    # Navigate through the JSON structure to find annotations
    projects = json_data.get('projects', {})
    for project_id, project_details in projects.items():
        labels = project_details.get('labels', [])
        for label in labels:
            annotations = label.get('annotations', {})
            objects = annotations.get('objects', [])
            for obj in objects:
                # Check if the object name matches the label we are interested in
                if obj.get('name') == label_name:
                    bounding_box = obj.get('bounding_box', {})
                    bounding_boxes.append(bounding_box)
    return bounding_boxes


# Load the contents of the JSON annotations file
print("[INFO] loading dataset...")
with open("/Users/eishahemchand/Mineral-Intelligence/output.json", "r") as file:  # Replace "annotations.json" with your JSON file's name
    annotations = json.load(file)

# Initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
targets = []
filenames = []
bounding_boxes = []

# Loop over the annotations
for annotation in annotations:
    filenames.append(annotation["data_row"]["external_id"])

    bounding_boxes.append(extract_bounding_box(annotation, "header"))

    #extract_bounding_box(annotation, "header")

print(filenames)
print(bounding_boxes)



final_json_list = []

for i in range(len(filenames)):
    filename = filenames[i]
    if len(bounding_boxes[i] == 0):
         boxObj = {"filename": filename, "startx": 0, "starty": 0, "endx": 0, "endy": 0}
         final_json_list.append(boxObj)

    else:
        print(bounding_boxes[i])
        box = bounding_boxes[i][0]
        print(box)

        top = box['top']
        left = box['left']
        height = box['height']
        width = box['width']

        startx = left
        starty = top
        endx = startx + width
        endy = starty + height
        boxObj = {"filename": filename, "startx": startx, "starty": starty, "endx": endx, "endy": endy}
        final_json_list.append(boxObj)



    
    






json_array_final = json.dumps(final_json_list, indent=4)

# # Optionally, write the JSON array to a new JSON file
with open('final_output.json', 'w') as json_file:
    json_file.write(json_array_final)







    








