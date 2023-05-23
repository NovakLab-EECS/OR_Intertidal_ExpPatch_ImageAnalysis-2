import torch
import glob
import os, os.path

model = torch.hub.load('ultralytics/yolov3', 'custom', path='runs/train/tues03083/weights/best.pt')

dir = '/nfs/hpc/share/mccallea/OR_Intertidal_ExpPatch_ImageAnalysis/ExpPatch-Pics/ExpPatchPics-Processed/2014-01-29_Survey_06_P'

images = []
for f in os.listdir(dir):
    ext = os.path.splitext(f)[1]
    if ext.lower() != '.jpg':
        continue
    images.append(os.path.join(dir, f))

print(images)

results = model(images)

results.print()
results.save()
results.show()

