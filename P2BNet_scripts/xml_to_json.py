import os
import re
import xml.etree.ElementTree as ET
import json
from pathlib import Path


def parse_xml(xml_path, jpg_path, type_mapping):
   tree = ET.parse(xml_path)
   root = tree.getroot()
   keypoints = []


   for marker_type in root.findall('Marker_Data/Marker_Type'):
       class_type = int(marker_type.find('Type').text)
       if class_type in type_mapping:
           for marker in marker_type.findall('Marker'):
               x = int(marker.find('MarkerX').text)
               y = int(marker.find('MarkerY').text)
               keypoints.append({'x': x, 'y': y, 'class': type_mapping[class_type]})
       else:
           print(f"Warning: Type {class_type} not found in type_mapping")


   record = {
       'image_path': jpg_path,
       'keypoints': keypoints
   }

   return record



def convert_to_json_format(dataset_dir, type_mapping):
   dataset = []
   pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_Survey_\d{2}_P")


   for root, _, files in os.walk(dataset_dir):
       if pattern.search(os.path.basename(root)):
           print(f'Processing directory: {root}')
          
           xml_files = [f for f in files if f.startswith('CellCounter_') and f.endswith('.xml')]
           jpg_files = [f for f in files if f.endswith('.jpg')]


           for xml_file in xml_files:
               base_name = xml_file[12:-4]
               jpg_file = f'{base_name}.jpg'
              
               if jpg_file in jpg_files:
                   xml_path = os.path.join(root, xml_file)
                   jpg_path = os.path.join(root, jpg_file)
                   record = parse_xml(xml_path, jpg_path, type_mapping)
                  
                   if record:
                       dataset.append(record)
                       print(f'Processed: {xml_file}')
                   else:
                       print(f'Skipped: {xml_file}')
               else:
                   print(f'No matching JPG file found for: {xml_file}')


   return dataset




if __name__ == '__main__':
   dataset_dir = './OR_Intertidal_ExpPatch_ImageAnalysis-2/ExpPatch-Pics/ExpPatchPics-Processed/'
   output_json = 'converted_dataset.json'


   type_mapping = {
       1: 'Bg', 2: 'BgD', 3: 'Cth', 4: 'CthD', 5: 'Sc', 6: 'ScD', 7: 'Myt', 8: 'MytD',
       9: 'No', 10: 'Nc', 11: 'Pp', 12: 'Lim', 13: 'Ls', 14: 'Ae', 15: 'Mt', 16: 'MtD',
       17: 'Mc', 18: 'McD', 19: 'Alg', 20: 'Lep', 21: 'Pis', 22: 'Chi', 23: 'Emp',
       24: 'Ner', 25: 'Wor', 26: 'Negg', 27: 'Sandtube_worm'
   }

   dataset = convert_to_json_format(dataset_dir, type_mapping)

   if dataset:
       with open(output_json, 'w') as f:
           json.dump(dataset, f, indent=2)
       print(f'Success: {len(dataset)} records written to {output_json}')
   else:
       print('Error: No records found. The output JSON file will not be created.')


