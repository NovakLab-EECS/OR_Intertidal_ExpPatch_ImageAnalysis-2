# Intro / Background
The main task that this project set out to accomplish was the identification and enumeration of all species present in a time series dataset of images taken by our project partner, Professor Mark Novak. A subset of the dataset images came coupled with .xml files containing x and y locations of each species in the picture, all of which were identified and labeled by hand. Each hand-labeled picture also came with a .xls file, which contained the total number of instances identified for each class type present in the corresponding picture.  Due to the massive amount of time it took to label even a small portion of the dataset by hand, we set out to train a machine learning model, using the hand-labeled data as a validation set, that could identify each instance of a species, and its correct type, in every picture in the entire time series dataset. We tried two different approaches for object detection, the first was a resnet pipeline and the second was a YOLO model. 

# The Dataset
The data set is composed of 38,88 images total. 1,944 of these images are labeled with points and species tags for each organism in the image. This subset was used for training as well as validation. Approximately 35 images were later taken from the training set and had some or all of the points in the image turned into bounding boxes around the organisms. This was done with a simple Python image labeling tool called [LabelImg](https://github.com/tzutalin/labelImg), which would automatically generate YOLO format text files containing the coordinates of our drawn bounding boxes. This dataset and the YOLO text files created from it were used to train and validate the YOLO models. For additional information about the dataset, including how it was collected, where and when it was collected and so fourth, refer to the README.md file in the main project repository. 

# Previously Attempted Strategies
Before moving towards our final strategy using Yolov5, we attempted the idea of using a simple “image analysis pipeline” using a pre-trained Resnet50 model. All source code related to this method can be found in the resnet_scripts folder. We started off by breaking all of the labeled images in our training set into tiles of varying sizes including (128, 128) and (256, 256). Each tile was of one instance of a single species. The borders for said tiles were calculated using the x, y coordinates for the specific instance to be captured. These coordinates were extracted by parsing and iterating through the xml file corresponding to the current image. After iterating over all of the images in the testing set, the result would be a series of folders, one for each class type, each of which contained a series of small tile images of an instance of that species. All the tile image folders were organized into one main folder.
  
Label and feature vectors were then extracted from our tiles directory using pytorch. The pretrained model that was used for this approach was a resnet50 model, loaded with the following line:
```
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)  
```
After setting up data transforms for the tiles, they were passed to pytorch’s data loader via the following lines:
```
train_datasets = datasets.ImageFolder(root=data_dir,transform=data_transforms['train'])

train_dataloader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

```
The label and feature vectors were stored as np arrays. After these np arrays were created, they were passed to a classifier to make predictions. The classifier used in this experiment was the sklearn MLP classifier. Creating an instance of the MLP classifier and utilizing it to make predictions was handled with the following lines:
```
clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(4096), learning_rate='adaptive', max_iter=10000)

clf.fit(features_np[0:MAX_ITERATION-1], labels_np[0:MAX_ITERATION-1])

predicted_label = clf.predict([features_np[MAX_ITERATION-1]])
actual_label = labels_np[MAX_ITERATION-1]
print(f"Predicted_label: {predicted_label}  Actual_label {actual_label}")
```
This process did not yield desirable results since the tiling method ended up producing too many no-class boxes (tiles with no species in them). The over saturation of no-class boxes led to high prediction accuracies overall but often incorrect predictions for tiles with actual species in them. We tried various methods to fix the class imbalance problem, such as resampling the minority classes and adjusting the loss function, but were unsuccessful in achieving high accuracies for the species of interest.
  
# Yolov5 Approach
YOLO requires bounding boxes for the objects it is trying to classify, so our team worked on manually labeling each instance of a species with a bounding box based on the center xy coordinate points provided to us. We ended up with approximately 35 images of partially to fully labeled bounding boxes to train our [YOLOv5](https://github.com/ultralytics/yolov5) model on. With our training set, we ran the YOLO model using the HPC server (how to connect to the HPC server is explained in the next section). Our set up for running the model on the servers required installing all YOLO dependencies into a python virtual environment, activating tmux so we can later detach from the session without interrupting the run, and requesting a GPU on the server with the following command:   
```  
srun -A cs462 -p share -c 2 -n 2 --gres=gpu:1 -t 2-00:00:00 --pty bash
```
Once the server has allocated us a GPU, we run the following command next:
Note: this command must be run from the “yolov5” directory 
```
python3 train.py --epochs 1500 --data dataset.yaml --weights yolov5m.pt --cache --freeze 10 --img 1280 --batch 2 --name [INSERT NAME OF RUN]  
```
An alternative method to train YOLO would be to create a bash script and run it without it being attached to a running process so there is no risk of the job being killed.
Example bash script:
  
train_yolo.bash
```
#!/bin/bash
#SBATCH -J train
#SBATCH -A cs462
#SBATCH -p share
#SBATCH -c 2
#SBATCH -n 2
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<youremail@oregonstate.edu>

source "<path to  venv bin/activate>"
python3 train.py --epochs ${1} --data dataset.yaml --weights yolov5m.pt --cache --freeze 10 --img 1280 --batch 2 --name ${2}  
```
This script can be run with the command:
```
sbatch train_yolo.bash <# of epochs> <run name>  
```
  
From the model runs we ran, we found the optimal parameters to be 10 layers for freezing, 1280 for image size, 2 for batch size, and 1500 for number of epochs to run. Since our dataset of labeled bounding boxes was insufficient for our desired accuracy goal, we needed to automate the process of further increasing our training data. We started working on a semi-supervised learning approach using YOLO. The process involves using YOLO to infer bounding boxes on new data, then our Python script will verify whether or not the bounding box contains the right class and if its size fits the corresponding class correctly. If the bounding box is correct, it is added into the training set.  
# Connecting to HPC Server
A quick start guide for getting set up on the HPC server can be found [here](https://it.engineering.oregonstate.edu/hpc/quick-start). In order to get access, email Robert Brian Yelle, yellero@oregonstate.edu. Additional storage space is likely required and a 1TB global HPC scratch directory can be requested when requesting permission for the HPC server. Once permission is granted, you will be assigned an account to access the HPC cluster, in our case it is ‘cs462 '.  


# Next Steps For Future Students
Ideally, we would like to have higher accuracy, at least on the species that we have a lot of data for. The species of particular interest in this data set that we are trying to infer relationships between are types 1, 7, 11, and 15. The more bounding boxes that we can draw for these species, the more accurate our model will likely be. There are 2 ways this could be done. 
  
The first is the way that we labeled most of our bounding boxes, which is using a python script to generate the xy coordinates over the corresponding image, and then using [LabelImg](https://github.com/tzutalin/labelImg), a python library to create bounding boxes in the same places with another monitor. This requires a lot of looking back and forth, and is very time consuming (some images have hundreds of points very close together, and we are not the best at differentiating, being computer science majors).  
  
The second way to create bounding boxes requires semi-supervised learning, where we give YOLO our current data set, and have it give us bounding boxes that it has predicted to add to our training set. With this approach, a script can be written to display these bounding boxes, then confirm or deny if the bounding box is accurate. This would expedite the process of adding training data, and is much less time consuming. 
  
We have a small subset of our total training data labeled with bounding boxes and the rest of the training data having x y coordinates for labels. We had a couple of ideas for adding more bounding boxes. The first would be to manually add more, but this is very time intensive. Another method would be to use a ‘semi-supervised’ algorithm to add more bounding boxes. We are doing this by running inference on the training data without bounding boxes, then using the x y coordinates present in the training data and checking if a point is in a bounding box and if it is and it is the same type, then add it to the training data.

CB- Test
