import scipy.io as scio
import numpy as np
import os
import json
from PIL import Image


def fmt_coco_annotation(bbox, cid, anno_id, img_id, iscrowd=0, ignore=None):
   x1, y1, x2, y2 = bbox
   ann = {'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
           'bbox': [x1, y1, x2 - x1, y2 - y1],
           'category_id': cid,
           'area': (y2 - y1) * (x2 - x1),
           'iscrowd': iscrowd,
           'image_id': img_id,
           'id': anno_id}


   if ignore is not None:
       ann['ignore'] = ignore
   return ann


def _to_pseudo_box(pts, wh=(15, 15)):
   #     wh = np.array(wh)
   #     x1y1 = pts - wh / 2
   #     x2y2 = pts + wh / 2
   #     WH = np.zeros(pts.shape) + wh
   #     return np.concatenate((x1y1, WH), axis=1)
   wh = np.array(wh)
   x1y1 = pts - wh / 2
   x2y2 = pts + wh / 2
   return np.concatenate((x1y1, x2y2), axis=1)


def _parse_json(entry, img_id, anno_id, categories):
   keypoints = entry['keypoints']
   bboxs = _to_pseudo_box(np.array([[kp['x'], kp['y']] for kp in keypoints]))
   annos, anno_id = _to_annos(anno_id, img_id, bboxs, [kp['class'] for kp in keypoints], categories)
   return annos, anno_id


def _to_annos(anno_id, img_id, bboxs, classes, categories):
   annos = []
   class_to_id = {cat['name']: cat['id'] for cat in categories}
   for box, cls in zip(bboxs, classes):
       category_id = class_to_id[cls]
       ann = fmt_coco_annotation(box, category_id, anno_id, img_id)
       annos.append(ann)
       anno_id += 1
   return annos, anno_id


def _fmt_image(file_name, height, width, img_id):
   return {'file_name': file_name,
           'height': height,
           'width': width,
           'id': img_id}


def generate_coco_fmt(annotations_file, categories):
   img_id, anno_id = 0, 0
   all_annos, all_images = [], []


   with open(annotations_file, 'r') as f:
       dataset = json.load(f)


   for entry in dataset:
       annos, anno_id = _parse_json(entry, img_id, anno_id, categories)
       all_annos.extend(annos)


       img = Image.open(entry['image_path'])
       all_images.append(_fmt_image(os.path.basename(entry['image_path']), img.height, img.width, img_id))
       img_id += 1


   return {
       "images": all_images,
       "annotations": all_annos,
       "categories": categories,
       "type": "instance"
   }


if __name__ == '__main__':


   categories = [    {'id': 1, 'name': 'Bg', 'supercategory': 'object'},
                     {'id': 2, 'name': 'BgD', 'supercategory': 'object'},
                     {'id': 3, 'name': 'Cth', 'supercategory': 'object'},
                     {'id': 4, 'name': 'CthD', 'supercategory': 'object'},   
                     {'id': 5, 'name': 'Sc', 'supercategory': 'object'},   
                     {'id': 6, 'name': 'ScD', 'supercategory': 'object'},   
                     {'id': 7, 'name': 'Myt', 'supercategory': 'object'},   
                     {'id': 8, 'name': 'MytD', 'supercategory': 'object'},   
                     {'id': 9, 'name': 'No', 'supercategory': 'object'},   
                     {'id': 10, 'name': 'Nc', 'supercategory': 'object'},   
                     {'id': 11, 'name': 'Pp', 'supercategory': 'object'},   
                     {'id': 12, 'name': 'Lim', 'supercategory': 'object'},   
                     {'id': 13, 'name': 'Ls', 'supercategory': 'object'},   
                     {'id': 14, 'name': 'Ae', 'supercategory': 'object'},   
                     {'id': 15, 'name': 'Mt', 'supercategory': 'object'},   
                     {'id': 16, 'name': 'MtD', 'supercategory': 'object'},   
                     {'id': 17, 'name': 'Mc', 'supercategory': 'object'},   
                     {'id': 18, 'name': 'McD', 'supercategory': 'object'},   
                     {'id': 19, 'name': 'Alg', 'supercategory': 'object'},   
                     {'id': 20, 'name': 'Lep', 'supercategory': 'object'},   
                     {'id': 21, 'name': 'Pis', 'supercategory': 'object'},   
                     {'id': 22, 'name': 'Chi', 'supercategory': 'object'},   
                     {'id': 23, 'name': 'Emp', 'supercategory': 'object'},   
                     {'id': 24, 'name': 'Ner', 'supercategory': 'object'},   
                     {'id': 25, 'name': 'Wor', 'supercategory': 'object'},   
                     {'id': 26, 'name': 'Negg', 'supercategory': 'object'},   
                     {'id': 27, 'name': 'Sandtube_worm', 'supercategory': 'object'}
               ]


   annotations_file = './converted_dataset.json'
   output_json = 'coco_annotations.json'
   coco_fmt_data = generate_coco_fmt(annotations_file, categories)
   if coco_fmt_data:
       with open(output_json, 'w') as f:
           json.dump(coco_fmt_data, f, indent=2)
       print(f'Success: {len(coco_fmt_data)} records written to {output_json}')
   else:
       print('Error: No records found. The output JSON file will not be created.')


