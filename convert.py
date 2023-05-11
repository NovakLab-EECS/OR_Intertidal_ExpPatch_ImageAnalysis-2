# Ashwin Subramanian
# Capstone
import xml.etree.ElementTree as ET  
import os
import cv2
import numpy as np
import json


categories_list=[{'category_id': 1, 'name': 'Balanus_glandula'},
                    {'category_id': 2, 'name': 'Balanus_glandula_(dead)'},
                    {'category_id': 3, 'name': 'Chthamalus_dalli'},
                    {'category_id': 4, 'name': 'Chthamalus_dalli (dead)'},
                    {'category_id': 5, 'name': 'Semibalanus_cariosus'},
                    {'category_id': 6, 'name': 'Semibalanus_cariosus_(dead)'},
                    {'category_id': 7, 'name': 'Mytilus_spp,Myt'},
                    {'category_id': 8, 'name': 'Mytilus_spp_(dead)'},
                    {'category_id': 9, 'name': 'Nucella_ostrina'},
                    {'category_id': 10, 'name': 'Nucella_canaliculata'},
                    {'category_id': 11, 'name': 'Pollicipes_polymerus'},
                    {'category_id': 12, 'name': 'Limpets'},
                    {'category_id': 13, 'name': 'Littorina_sitkana'},
                    {'category_id': 14, 'name': 'Anemones'},
                    {'category_id': 15, 'name': 'Mytilus_trossulus'},
                    {'category_id': 16, 'name': 'Mytilus_trossulus_(dead)'},
                    {'category_id': 17, 'name': 'Mytilus_californianus'},
                    {'category_id': 18, 'name': 'Mytilus_californianus_(dead)'},
                    {'category_id': 19, 'name': 'Algae'},
                    {'category_id': 20, 'name': 'Leptasterias'},
                    {'category_id': 21, 'name': 'Pisaster'},
                    {'category_id': 22, 'name': 'Chiton_spp'},
                    {'category_id': 23, 'name': 'Emplectonema worm'},
                    {'category_id': 24, 'name': 'Nereis_worm'},
                    {'category_id': 25, 'name': 'Other_worm spp.'},
                    {'category_id': 26, 'name': 'Nucella_eggs'},
                    {'category_id': 27, 'name': 'Sandtube_worm'}
                    ]


def parseXML(imageXMLPath):
    tree = ET.parse(imageXMLPath)

    root = tree.getroot()
    image_name = root[0][0].text
    marker_data = root[1][1:]
    
    type_locs = dict()
    for marker_type in marker_data:
       
        type = marker_type[0].text
        image_locs = np.array([[int(marker[0].text), int(marker[1].text)] for marker in marker_type.findall('Marker')], dtype = int)
        type_locs[type] = image_locs
        
    return (image_name, type_locs)

 
def convert_to_quasi_center_annotations(coords, image_id): 
    # Load the data,
    x = []
    y = []
    class_labels = []
    for cx,cy in coords:
        x.append(int(cx))
        y.append(int(cy))
        class_labels.append(image_id) #this is under the assumption that each image contains 1 of the same species


    # Calculate the bounding boxes
    bounding_boxes = []
    for i in range(len(x)):
        bounding_boxes.append([x[i]-48, y[i]-48, x[i]+48, y[i]+48])

    # Define the central ellipse
    central_ellipses = []
    for bbox in bounding_boxes:
        bx, by, bw, bh = bbox
        central_ellipses.append([bx, by, bw * 0.25, bh * 0.25])

    # Determine the intersecting area
    intersecting_areas = []
    for i in range(len(bounding_boxes)):
        bbox = bounding_boxes[i]
        ellipse = central_ellipses[i]
        # Assume the bounding box is the mask
        intersecting_areas.append(bbox)

    # Calculate the Rectified Gaussian Distribution
    rg_distributions = []
    for i in range(len(intersecting_areas)):
        area = intersecting_areas[i]
        # Assume µ=0 and σ=1/4 for simplicity
        mu = 0
        sigma = 1/4
        # Calculate the distribution
        x = np.linspace(area[0], area[2], 1000)
        y = np.linspace(area[1], area[3], 1000)
        X, Y = np.meshgrid(x, y)
        Z = (1/(2*np.pi*sigma**2)**0.5) * np.exp(-(X-mu)**2/(2*sigma**2) - (Y-mu)**2/(2*sigma**2))
        rg_distributions.append(Z)

    return bounding_boxes, central_ellipses, intersecting_areas, rg_distributions



if __name__ == "__main__":
    imagedir = 'OR_Intertidal_ExpPatch_ImageAnalysis/ExpPatch-Pics/ExpPatchPics-Processed/'

    image_dic = dict()
    imList = []
    annList = []
    catList = []
    
    imCounter = 0
    print("Collecting Images")
    for folder in os.listdir(imagedir):
        for file in os.listdir(imagedir + folder):
            if file.split('.')[-1] != 'JPG':
                continue

            image_dic[file] = imCounter
            imCounter += 1

            image = cv2.imread(imagedir + folder + '/' + file)
            height, width = image.shape[:2]

            im_dic = {
                "id":imCounter,
                "width":width,
                "height":height,
                "file_name": file,
                "license":1,
                "date_captured":""
            }

            imList.append(im_dic)
 
    annCounter = 0 
    print("Generating Bounding Boxes")
    # d = open("file.json", 'w')
    c = 0
    for folder in os.listdir(imagedir):
        for file in os.listdir(imagedir + folder):
            print(c)
            c += 1
            if file.split('.')[-1] != 'xml':
                continue
            
            image_locsPath = imagedir + folder + '/' + file
        
            image_loc_dic = dict()
            out = parseXML(image_locsPath)
            image_name = out[0].split(' ')[-1]

            image_id = 0
            if image_name not in image_dic.keys():
                image_id = max(image_dic.values()) + 1
                image_dic[image_name] = image_id
            else:
                image_id = image_dic[image_name]
            
            image_loc_dic[image_id] = out[1]

            for key in image_loc_dic[image_id]:
                bounding_boxes, central_ellipses, intersecting_areas, rg_distributions = convert_to_quasi_center_annotations(image_loc_dic[image_id][key], image_id)         
                for i in range(len(bounding_boxes)):

                    bbox = bounding_boxes[i]
                    ellipse = central_ellipses[i]
                    area = intersecting_areas[i]
                    rg_distribution = rg_distributions[i]

                    ann = {
                    "id": annCounter,
                    "image_id": image_id,
                    "category_id": key,
                    "bbox": bbox,
                    "ellipse": ellipse,
                    "area": area,
                    "rg_distribution": rg_distribution,
                    "iscrowd": 0,
                    "segmentation": []
                    }

                    # d.write(str(ann))
                    annList.append(ann)
                    annCounter += 1
            
    print("Done generating annotations")

    for i in range(len(categories_list)):
        cat = {
        "id": categories_list[i]["category_id"],
        "name": categories_list[i]["name"],
        "supercategory":"none"
        }
        catList.append(cat)


    json_obj = { 
        "images" : imList,
        "annotations" : annList,
        "categories" : catList
        }


    json_file = json.dumps(json_obj, indent=4)

    print("Writing to JSON file")
    # Writing to sample.json
    with open("coco.json", "w") as outfile:
        # outfile.write('{')
        outfile.write(json_file)
    
    print("done")


