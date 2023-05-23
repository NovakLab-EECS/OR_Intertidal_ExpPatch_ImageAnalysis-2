import os
import subprocess
import cv2
import matplotlib.pyplot as plt
from unicodedata import normalize
import re
from mysuperawesomexml import get_points

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

# subfolders = [f.path for f in os.scandir(SRC_DIR) if f.is_dir()]
subfolders = ['/nfs/hpc/share/mccallea/OR_Intertidal_ExpPatch_ImageAnalysis/ExpPatch-Pics/ExpPatchPics-Processed/a']


exp_folders = []
img_folders = []

for subfolder in subfolders:
  print("Starting folder: " + subfolder)
  # os.system(f"python detect.py --source {subfolder} --weights ./runs/train/tues03083/weights/best.pt --save-txt --save-crop")

  # out = subprocess.check_output(f"python detect.py --source {subfolder} --weights ./runs/train/tues03083/weights/best.pt --save-txt --save-crop", shell=True)
  out = subprocess.Popen(f"python detect.py --source {subfolder} --weights ./runs/train/tues03083/weights/best.pt --save-txt --save-crop", shell=True, stderr=subprocess.PIPE).stderr
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
  break


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

    # Get the path to the image file
    image = os.path.join(img_folders[idx], image_file)

    # The file name is in the format of "CellCounter_{Image Name}.xml"
    xml_filename = f"CellCounter_{image_file.split('.')[0]}.xml"
    xml_file = os.path.join(img_folders[idx], xml_filename)
    points = get_points(xml_file)

    # Get the shape of the image
    img = cv2.imread(image)
    dh, dw, _ = img.shape

    print(f"Image Shape: {dh}x{dw}")

    # Read each line in the text file
    with open(os.path.join(labels_path, text_file), 'r') as f:
      lines = f.readlines()

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

        print(f"{image_file} {l} {t} {r} {b}")


        # Check if the bounding box contains some point
        containsPoint = False

        # Iterate over all the points to check if they are within the bounding box
        for point in points:
          pClass = point['type']
          pCoord = point['coord']

          x = pCoord[0]
          y = pCoord[1]

          # Check to see if its within the bounding box
          if x > r or x < l or y > b or y < t:
            continue

          # If it is within the bounding box, then we want to check the types
          if pClass != type:
            continue

          # We know the point is within the boundingbox and the type is the same
          # We want to add to the training set
          print(f"{image_file} {x} {y}")
          containsPoint = True

        print(containsPoint)

        # If the bounding box contains a point, then we want to add it to the training set
        if containsPoint:
          # We now want to add the image to the training set
          # To do this,
          # 1. Copy the image to the training set images folder
          # 2. Upsert the bounding box to the training set labels folder

          # Get the path to the image file
          image_path = os.path.join(img_folders[idx], image_file)
          print(image_path)


