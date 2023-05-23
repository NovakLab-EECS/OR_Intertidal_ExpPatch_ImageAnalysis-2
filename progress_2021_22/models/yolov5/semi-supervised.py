from enum import unique
import os
import subprocess
import cv2
import matplotlib.pyplot as plt
from unicodedata import normalize
import re
from mysuperawesomexml import get_points
import numpy as np

def escape_ansi(text):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', text)

# Command we want to run is
# python detect.py --source  ~/hpc-share/OR_Intertidal_ExpPatch_ImageAnalysis/ExpPatch-Pics/ExpPatchPics-Processed/{YYYY-MM-DD_Survey_XX_P}/ --weights ./runs/train/tues03083/weights/best.pt --save-txt --save-crop

SRC_DIR = '/nfs/hpc/share/mccallea/OR_Intertidal_ExpPatch_ImageAnalysis/ExpPatch-Pics/ExpPatchPics-Processed/'
IMG_DIR = '/nfs/stak/users/mccallea/hpc-share/OR_Intertidal_ExpPatch_ImageAnalysis/ExpPatch-Pics/ExpPatchPics-Processed/'
DATASET_DIR = '/nfs/stak/users/mccallea/hpc-share/cs462/yolo/datasets/species'

# Yolo format for the bounding boxes
# All x/y/w/l values are normalized to the len/wid of the image
# obj_class x y w h

#STRATEGY
# run inference on every folder and grab that data
# We want to get all the folder names in the source directory
# and then run the command for each folder
# for each bounding box location file scale the coordinates to the image
# get all the xy coordinates for the species type from the coresponding image's xml file
# check to see if any bounding boxes are around any of those points
# if yes, add that bounding box to the image, if no then don't
# end result should be: list of bounding boxes that are true positives, which we then put in the training data set

subfolders = [f.path for f in os.scandir(SRC_DIR) if f.is_dir()]
# subfolders = ['/nfs/hpc/share/mccallea/OR_Intertidal_ExpPatch_ImageAnalysis/ExpPatch-Pics/ExpPatchPics-Processed/a']

print(f"Got {len(subfolders)} subfolders")

max_iters = 3 
curr_iters = 0

exp_folders = []
img_folders = []

for subfolder in subfolders:
  if curr_iters > max_iters:
    print(f"Max iterations reached, breaking out of loop")
    break
  curr_iters += 1

  print("Starting folder: " + subfolder)
  # os.system(f"python detect.py --source {subfolder} --weights ./runs/train/tues03083/weights/best.pt --save-txt --save-crop")

  # out = subprocess.check_output(f"python detect.py --source {subfolder} --weights ./runs/train/tues03083/weights/best.pt --save-txt --save-crop", shell=True)
  out = subprocess.Popen(f"python detect.py --source {subfolder} --weights ./weights/best.pt --save-txt --save-crop", shell=True, stderr=subprocess.PIPE).stderr
  out = out.read()

  # Convert out from bytes to string
  out = out.decode('utf-8')

  # Split the string by newline
  out = out.split('\n')

  results_path = None

  # Look for the string which indicates where the results are saved to
  # and then get the path to the results file
  for line in out:
    if 'Results saved to ' in line:
      line = escape_ansi(line)
      results_path = line.split('Results saved to ')[1]
      break

  if results_path is None:
    print("No results path found")
    continue

  # The command will print something along the lines of "Results saved to runs/detect/exp{N}"
  # We want to get that number and save it to an array
  exp_folder = results_path.split('/')[-1]
  exp_folders.append(exp_folder)
  img_folders.append(subfolder)

  print("Finished folder: " + subfolder)


# Now we have the exp_folders array. We want to get the bounding boxes for each of the images.
# We can do this by iterating over the text files in the labels folder
for idx, exp_folder in enumerate(exp_folders):
  print("Starting exp folder: " + exp_folder)
  # Get the path to the labels folder
  labels_path = os.path.join('./runs/detect/', exp_folder, 'labels')

  # If the labels folder doesn't exist, skip this run
  # If the labels folder does exist but is empty, skip this run
  if not os.path.exists(labels_path):
    print("No labels folder found")
    continue
  elif len(os.listdir(labels_path)) == 0:
    print("Labels folder is empty")
    continue

  # Get all the text files in the labels folder
  text_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
  print(f"Text Files: {text_files}")

  # For each text file in the labels folder
  for text_file in text_files:
    # Get the corresponding image file
    image_file = text_file.split('.')[0] + '.JPG'

    print('\n\n=========================================================')
    print('Image File: ' + image_file.split('.')[0])


    # Get the path to the image file
    image = os.path.join(img_folders[idx], image_file)

    # The file name is in the format of "CellCounter_{Image Name}.xml"
    xml_filename = f"CellCounter_{image_file.split('.')[0]}.xml"
    xml_file = os.path.join(img_folders[idx], xml_filename)
    points = get_points(xml_file)

    if points is None:
      print("No points found")
      print('=========================================================')
      continue

    # Get the amount of unique classes in the points array
    unique_classes = set([point['type'] for point in points])
    unique_classes = list(unique_classes)
    print(f"Unique Classes: {unique_classes}")

    # Generate a unique color for each class
    colors = []
    for i in range(len(unique_classes)):
      color: tuple = np.random.randint(0, 255, size=(3, ))
      #convert data types int64 to int
      color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
      colors.append(color)


    # Get the shape of the image
    img = cv2.imread(image)

    if img is None:
      image_file = text_file.split('.')[0] + '.jpg'
      image = os.path.join(img_folders[idx], image_file)
      print(f"Image file not found, trying jpg with {image_file}")
      img = cv2.imread(image)
      if img is None:
        print("Image file not found")
        print('=========================================================')
        continue

    dh, dw, _ = img.shape

    print(f"Image Shape: {dh}x{dw}")

    # Read each line in the text file
    with open(os.path.join(labels_path, text_file), 'r') as f:
      lines = f.readlines()

      print(f"Total Lines: {len(lines)}")

      # For each line in the text file
      for line in lines:
        # Each line is structured as follows:
        # class x y w h

        # Reference: https://stackoverflow.com/a/64097592

        # Split string to float
        type, x, y, w, h = map(float, line.split())

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        # Error check
        if l < 0:
          l = 0
        if r > dw - 1:
          r = dw - 1
        if t < 0:
          t = 0
        if b > dh - 1:
          b = dh - 1

        # Check if the bounding box contains some point
        containsPoint = False
        pointsContained = []

        # Iterate over all the points to check if they are within the bounding box
        for point in points:
          pClass = point['type']
          pCoord = point['coord']

          x: int = pCoord[0]
          y: int = pCoord[1]
          
          # Get the color for the class
          color = colors[unique_classes.index(pClass)]
          
          # Draw a circle around the point
          img = cv2.circle(img, (x, y), 15, color, 5)

          # Check to see if its within the bounding box
          if x > r or x < l or y > b or y < t:
            continue

          # If it is within the bounding box, then we want to check the types
          if pClass != type:
            continue

          # We know the point is within the boundingbox and the type is the same
          # We want to add to the training set

          # Draw the bounding box
          img = cv2.rectangle(img, (l, t), (r, b), color, 5)

          print(f"Drawing the bounding box for ")

          containsPoint = True
          pointsContained.append(point)

        # If the bounding box contains a point, then we want to add it to the training set
        if containsPoint:
          print("We have {} points in the bounding box".format(len(pointsContained)))
          
          # We now want to add the image to the training set
          # To do this,
          # 1. Copy the image to the training set images folder
          # 2. Upsert the bounding box to the training set labels folder

          # Get the path to the image file
          # image_path = os.path.join(img_folders[idx], image_file)
          # print(image_path)

          # Save the image
    # Save the img 
    cv2.imwrite(os.path.join('./debug', image_file), img)
    print('=========================================================')


