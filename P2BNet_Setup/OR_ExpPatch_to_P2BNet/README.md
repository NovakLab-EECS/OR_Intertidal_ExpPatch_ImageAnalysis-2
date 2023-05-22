# P2BNet README

This guide explains how to use the P2BNet machine learning model with the "ExpPatchPics-Processed" dataset.

## Prerequisites

To follow this guide, you must have:

- Conda package manager installed
- GPU access

## Getting Started

1. **Create a new conda environment.**

   Navigate to the top level of the P2BNet directory and create a new conda environment using the `environment.yml` file:

    conda env create -f environment.yml

    Activate the environment:

    conda activate <env_name>


2. **Convert XML annotation files to a single JSON annotation file.**

    Run the `xml_to_json.py` script. You can customize the dataset directory path by changing the `dataset_dir` variable in the script. Likewise, you can specify the name and location of the output JSON file by modifying the `output_json` variable.

    python xml_to_json.py


3. **Create a COCO dataset style annotation file.**

    Execute the `json_to_coco.py` script. To do this, modify the `annotations_file` variable to the path of the JSON annotations and set the `output_json` path to the desired location and name of the new COCO style annotations file.

    python json_to_coco.py


4. **Build a new COCO style data directory.**

    Run the `build_data_dir.py` script. In this script, set the `src_dir` variable to the path where the data images are located. Adjust the `annotations_file` to the path where the COCO style annotation file is found. Set the `output_base_dir` to the directory where the new data directory should be stored.


    python build_data_dir.py


    The new data directory should be placed in the `P2BNet/TOV_mmdetections/data` directory to be used with the P2BNet model. If you're creating a custom configuration file, you can create a subdirectory in the `P2BNet/TOV_mmdetections/data` directory that matches your dataset or simply name it "custom". If you're using a predefined config file, the data subdirectory should be named "coco".

5. **Convert COCO style image annotations to P2BNet style.**

    Run the `coco_to_quasi.sh` shell script in the `P2BNet/TOV_mmdection` directory. This will convert the COCO style image annotations to P2BNet's "quasi center point annotations".

    sh coco_to_quasi.sh


6. **Produce P2BNet true bounding boxes.**

    Execute the `bbox_to_true_bbox.sh` script, also in the `P2BNet/TOV_mmdection` directory. This script will convert bounding boxes to the true bounding boxes format required by P2BNet.

    sh bbox_to_true_bbox.sh


7. **Train the model.**

    Finally, to start training your model, run the `train.sh` script in the `P2BNet/TOV_mmdetection` directory.


Congratulations, you've successfully prepared your data and are ready to use the P2BNet model with the "ExpPatchPics-Processed" dataset!

For additional assistance or information, please refer to the [official P2BNet GitHub repository](https://github.com/ucas-vg/P2BNet/tree/main/TOV_mmdetection) and its README and tutorials.





