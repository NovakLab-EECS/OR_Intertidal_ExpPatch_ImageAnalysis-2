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


# Data Analysis
  
# P2B Approach

P2BNet is a novel object detection model that was introduced in the paper "P2BNet: Point-to-Box Network for Object Detection" by Zhiqiang Shen, Jiaxiong Qiu, Weiming Dong, and Nong Sang. The main idea behind P2BNet is to convert the problem of object detection from a point-based prediction task to a box-based prediction task.

The model is a two-stage architecture, similar to that of the popular Faster R-CNN model. In the first stage, P2BNet uses a backbone network, such as ResNet or VGG, to extract features from the input image. These features are then fed into a Region Proposal Network (RPN) to generate a set of potential object regions, or "proposals."

In the second stage, P2BNet takes the proposals generated by the RPN and uses them to predict the final bounding boxes for the objects in the image. However, instead of predicting the bounding boxes directly, P2BNet uses a Point-to-Box (P2B) module to convert the predictions into a box-based format.

The P2B module is a novel component of P2BNet that is designed to address the limitations of point-based prediction methods. It takes as input a set of keypoints, or "points," that correspond to the locations of potential objects in the image, and generates a set of bounding boxes that tightly enclose these points.

The P2B module is implemented as a fully convolutional neural network (CNN) that takes the features extracted by the backbone network as input and produces a set of box predictions as output. The P2B module is trained end-to-end with the rest of the model, and uses a combination of convolutional layers, non-linear activation functions, and pooling layers to learn the mapping from points to boxes.

Finally, P2BNet uses a post-processing step to merge overlapping box predictions and remove duplicate detections, resulting in a final set of high-confidence object detections.

We replaced the previous teams YOLO approach with the P2BNet approach to benefit from its box-based prediction task, which we believe will provide better results for our specific problem of species identification in images.

# Model Refinement
As part of our efforts to improve the performance of our object detection model, Our team has layed out a process for model refinement and data augmentation. This plan outlines the steps we will undertake to ensure our model is more accurate and robust in detecting objects within various environments.

Data Augmentation:
a. Implement various augmentation techniques to increase the diversity and size of our dataset, such as:
    - Horizontal and vertical flipping
    - Rotation
    - Scaling and resizing
    - Shearing and skewing
    - Random cropping
    - Brightness and contrast adjustments
    - Noise injection
b. Apply these augmentations in a balanced manner to avoid overfitting and ensure the model generalizes well to new, unseen data.
c. Use a data augmentation library like Albumentations, imgaug, or build custom augmentation functions to streamline the process.

Model Selection and Architecture Refinement:
a. Choose a suitable baseline model, such as YOLO, Faster R-CNN, or SSD, that has demonstrated strong performance alongside P2BNet.
b. Optimize the model architecture by adjusting parameters such as the number of layers, filter sizes, and activation functions.
c. Incorporate techniques like batch normalization, dropout, and skip connections to improve model performance and reduce overfitting.

Model Training:
a. Split the pre-processed and augmented dataset into training, validation, and testing sets.
b. Train the model using a suitable optimization algorithm (e.g., Adam, RMSprop) and loss function (e.g., cross-entropy loss, IoU loss).
c. Monitor the model's performance using metrics like mean average precision (mAP) and Intersection over Union (IoU) on the validation set during training.
d. Employ techniques like early stopping, learning rate scheduling, and model checkpointing to prevent overfitting and ensure the most accurate model is saved.

Model Evaluation and Hyperparameter Tuning:
a. Evaluate the trained model's performance on the test set to obtain an unbiased estimate of its accuracy and generalization ability.
b. Perform hyperparameter tuning using techniques like grid search, random search, or Bayesian optimization to further optimize the model's performance.
c. Re-train and evaluate the model with the optimal set of hyperparameters to maximize its object detection capabilities.

Model Deployment and Continuous Improvement:
a. Deploy the refined object detection model on the remaining un-annotated Serial Image Analysis data.
b. Monitor the model's performance on remaining serial image analysis data and gather feedback from project partner (Mark Novak) to identify areas for improvement.

By following this plan, we will be able to develop a more accurate and robust object detection model that can effectively detect and classify objects in a wide variety of scenarios. 
