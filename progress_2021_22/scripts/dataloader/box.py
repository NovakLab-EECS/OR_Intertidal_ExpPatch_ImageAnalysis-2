from gettext import find
import numpy as np
from PIL import Image
import os
from yolov5.utils.plots import Annotator, Colors
from collections import defaultdict, Counter

LABELS_DIR = "../species/labels/" # directory of yolo labels, need to change if not run in correct yolov5 directory
IMAGE_DIR = "../species/images/"  # directory of corresponding images, need to change if not run in correct yolov5 directory

im_width = 3968
im_height = 2976


#	define type for image and label structure
class ImageAndLabel:
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels

def getLabels():
    images = os.listdir(LABELS_DIR)
    return images


def getImages(labels):
    imagesAndLabels = []
    images = os.listdir(IMAGE_DIR)

    for image in images:
        for label in labels:
            #	if the image and label have the same name, create ImageAndLabel obj
            #	comparing the first 18 chars to ingore extension
            if(label[0:17] == image[0:17]):
                i = ImageAndLabel(IMAGE_DIR + image, LABELS_DIR + label)
                imagesAndLabels.append(i)
                labels.remove(label)
    return imagesAndLabels


# Show bounding boxes in a single image
def makeBox(im_file, label_file):
    colors = Colors()
    im = Image.open(im_file)
    f = open(label_file, 'r')
    labels = f.read()
    labels = [label.split() for label in labels.split('\n')][:-1]

    annotator = Annotator(np.ascontiguousarray(im), line_width=3)

    for i, (cls, x, y, w, h) in enumerate(labels):
        x, y, w, h = float(x), float(y), float(w), float(h)
        # Add bounding box
        annotator.box_label(
            ((x-w)*im_width, (y-h)*im_height, (x+w)*im_width, (y+h)*im_height, im),
            f"line: {i, cls, x, y, im_file}",
            color=colors(int(cls))
        )

    im = annotator.result()
    im = Image.fromarray(im)
    im.show()


# find average bounding box size per species, then remove bounding box data that is larger/smaller than the avg * factor size
def find_average(factor):
    labels = os.listdir(LABELS_DIR)
    sizes = defaultdict(lambda: 0)
    instances_counter = Counter()

    for label in labels:
        file_path = os.path.join(LABELS_DIR, label)
        with open(file_path) as f:
            for line in f:
                species, x, y, w, h = tuple([float(x) for x in line.split()])
                sizes[species] = sizes[species] + w * h
                instances_counter[species] += 1
    print(sizes, instances_counter)
    for key in sizes:
        sizes[key] = sizes[key] / float(instances_counter[key])

    for label in labels:
        file_path = os.path.join(LABELS_DIR, label)
        counter = 0
        with open(file_path, "r") as f:
            lines = f.readlines()
            f.seek(0)
            for line in lines:
                species, x, y, w, h = tuple([float(x) for x in line.split()])
                if w * h < sizes[species] * float(factor) and w * h > sizes[species] / float(factor): # valid data point, write back to file
                    f.write(line)
                else: 
                    counter += 1
            f.truncate()
            print(f"Removed {counter} data points from {label}.")



# Show bounding boxes in all images
def show_all(images_and_labels):
    for obj in images_and_labels:
        im_file = obj.image
        label_file = obj.labels
        makeBox(im_file, label_file)
        

images_and_labels = getImages(getLabels())
find_average(5) # removes all bounding boxes from data that is 5x larger or smaller than the avg for its species
show_all(images_and_labels)

