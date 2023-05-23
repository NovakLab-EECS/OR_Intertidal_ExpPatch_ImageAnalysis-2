import os
import json

"""
Get the highest number of the experiment
"""
def get_highest_exp_number(path: str = './experiments/') -> int:
  exp_number = 0
  for filename in os.listdir(path):
    if filename.startswith('exp'):
      # Convert string into an array of chars
      chars = list(filename)
      # Filter out non-numeric chars
      chars = [char for char in chars if char.isdigit()]

      # Theres an edge case where we just have 'exp' in the filename and no number
      # this results in an empty array and we need to check for that
      if len(chars) == 0:
        chars = ['0']

      # Get the number of the experiment by removing the 'exp' and then converting the rest of the chars into an int
      exp_number = max(exp_number, int(''.join(chars)))
  return exp_number


"""
  The goal of this script is to create a json file that contains the bounding boxes and the labels of the images.
  The text file will be stored under the directory "runs/detect/exp{XYZ}" where {XYZ} is the number of the experiment.
  The json file will be named "detections.json" and will be stored in the same directory.
"""
def main():
  # Get the highest number of the experiment
  exp_number = get_highest_exp_number('./runs/detect/')

  print('The highest number of the experiment is: {}'.format(exp_number))

  # Get the path to the experiment
  exp_path = './runs/detect/exp{}'.format(exp_number)

  # Print the path to the experiment
  print('The path to the experiment is: {}'.format(exp_path))

  """
  The predicted bounding boxes will be stored in a folder called "labels" in the experiment folder
  Each image will have a text file with the bounding boxes and the labels. We want to store each filename
  as the key for json and the bounding boxes and labels as the value.
  """

  # Get the path to the labels folder
  labels_path = os.path.join(exp_path, 'labels')

  # Create a dictionary to store the bounding boxes and labels
  detections = {}

  # Loop through all the files in the labels folder
  for filename in os.listdir(labels_path):
    # Get the path to the label file
    label_path = os.path.join(labels_path, filename)

    # Open the label file
    with open(label_path, 'r') as f:
      # Store the bounding boxes and labels in the dictionary

      lines = f.readlines()

      # Each line is the format of
      # <class> <x> <y> <width> <height>
      # We want to store the bounding boxes and labels in a list
      # Each element in the list is a dictionary with the bounding box and label

      # Create a list to store the bounding boxes and labels
      bounding_boxes = []

      for line in lines:
        # Split the line into an array of strings
        line = line.split()

        # Get the bounding box and label
        bounding_box = {
          'x': float(line[1]),
          'y': float(line[2]),
          'width': float(line[3]),
          'height': float(line[4]),
          'label': int(line[0])
        }

        # Append the bounding box and label to the list
        bounding_boxes.append(bounding_box)


      detections[filename.split('.')[0]] = bounding_boxes

  # Create the path to the json file
  json_path = os.path.join(exp_path, f"detections-{exp_number}.json")

  # Write the json file
  with open(json_path, 'w') as f:
    json_string = json.dumps(detections)
    f.write(json_string)



if __name__ == "__main__":
  main()
