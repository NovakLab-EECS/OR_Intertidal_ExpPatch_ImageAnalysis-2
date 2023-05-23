import copy
import pandas as pd
import logging
import numpy as np
import os
import re
import torch

from abc import abstractmethod
from os.path import dirname, abspath
from skimage import io
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Tuple, Optional, Union
from PIL import Image

import xml.etree.ElementTree as ET

logging.basicConfig(
    format="%(asctime)s [%(name)s: %(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("dataset.py")


class PatchDataset(Dataset):
    """Dataset class for patch data."""

    # Supported date formats: YYYY-MM-DD, YYYY_MM_DD
    # Specified here:
    # https://github.com/NovakLab-EECS/OR_Intertidal_ExpPatch_ImageAnalysis/blob/master/ExpPatch-Info/ExpPatch_PictureFileNameConventions.txt
    _DATE_FORMAT = "([0-9]{4}\-[0-9]{2}\-[0-9]{2})|([0-9]{4}\_[0-9]{2}\_[0-9]{2})"

    def __init__(
        self,
        dataset_path: Union[os.PathLike, str],
        x_transform: Optional[Callable] = None,
        y_transform: Optional[Callable] = None,
    ):
        """init method.

        Args:
            dataset_path: directory containing all of the files that contain the samples.
            transform: Optional transform to be applied on a sample.
        """
        self._dataset_path = dataset_path
        self._transform = x_transform
        self._y_transform = y_transform
        self._abs_path = dirname(dirname(dirname(abspath(__file__))))

    def _check_attr(self, attr_name: str, check_val_exists: bool = False):
        if not hasattr(self, attr_name):
            raise RuntimeError(
                f"{type(self).__name__} needs to call super().__init__() before {attr_name} can be used."
            )
        if check_val_exists:
            if getattr(self, attr_name) is None:
                raise RuntimeError(f"{attr_name} is not initialized")

    def __len__(self) -> int:
        self._check_attr('_df', check_val_exists=True)
        return len(self._df)

    def _search_for_file(self, **kwargs):
        """Generate a mapping from a given index to a relative file path.

        Args:
            file_type: file type of files that will be added to dataset.
        """
        buf = []
        columns = None
        self._check_attr("_dataset_path", check_val_exists=True)
        total_files = 0
        for root, dirs, files in os.walk(self._dataset_path, topdown=True):
            # for name in files:
            if files:
                total_files += len(files)
                grouped_files = self._map_files_to_patches(
                    # name=name,
                    root=root,
                    dirs=dirs,
                    files=files,
                    kwargs=kwargs,
                )
                if grouped_files is not None:
                    for grouped_file in grouped_files:
                        buf.append(grouped_file.values())
                    if columns is None:
                        columns = list(grouped_files[0].keys())
        self._df = pd.DataFrame(buf, columns=columns)

    @abstractmethod
    def _map_files_to_patches(self, **kwargs):
        """Method that defines a policy for filling patches"""
        pass

    def _read_class_counts(self, xls_file_path: Union[str, os.PathLike]):
        class_counts = pd.read_table(xls_file_path)

        # Extract Total counts of each type
        # Drop "Total" in first column
        class_counts = class_counts.to_numpy()[0][1:]
        return class_counts

    def _read_landmarks(self, xml_file_path: Union[str, os.PathLike]) -> Dict:
        """Method to read species coordinate data from .XML file
        Args:
            xml_file_path: Path to XML file containing species coordinate
        Returns:
        """
        with open(xml_file_path, "r") as f:
            data = f.read()

            # Import tree from XML string
            root = ET.fromstring(data)

            # Create an dictionary to store the coordinates for each type
            coords = {}

            # Iterate through each species
            for item in root.findall("Marker_Data/Marker_Type"):
                markerType = -1
                # Find the children elements of Marker_Type
                for child in item:

                    if child.tag == 'Type':
                        # Get the type of the species
                        type = child.text

                        # Find the index of the type
                        markerType = int(type)

                        # Create a list to store the coordinates
                        coords[markerType] = []

                    # If the child is a coordinate
                    if child.tag == "Marker":
                        # Build the coordinate
                        coord = [-1, -1]

                        for subchild in child:
                            if subchild.tag == "MarkerX":
                                # Add the x coordinate to the list
                                coord[0] = int(subchild.text)
                            elif subchild.tag == "MarkerY":
                                # Add the y coordinate to the list
                                coord[1] = int(subchild.text)

                        if coord[0] != -1 and coord[1] != -1:
                            # Add the coordinate to the list
                            coords[markerType].append(coord)

        # Return the coordinates
        return coords

    def to_file(self, file_name: Optional[str] = None):
        """Put dataset in a csv file

        Args:
            file_name: name of file to put it in
        """
        if file_name is None:
            file_name = "PatchDataset.csv"

        logger.info("Saving dataset to (%s)", file_name)

        self._df.to_csv(file_name, encoding="utf-8")

    @abstractmethod
    def __getitem__(self, idx):
        """getitem method."""
        pass


class PatchPicsDataset(PatchDataset):
    """Dataset class for .jpg images"""

    _FILE_TYPE = {
        ".jpg": 0,
        ".xml": 1,
        ".xls": 2,
    }

    def __init__(
        self,
        dataset_path: Union[os.PathLike, str],
        transform: Optional[Callable] = None,
    ):
        super().__init__(dataset_path, transform)
        self.f = []
        self._search_for_file()

    def _map_files_to_patches(
        self, root: str, dirs: List[str], files: List[str], **kwargs
    ):
        def _get_associated_files(_files, _tmp_len_files, _relative_path):
            # TODO: fix so it finds all of the files, ending condition stops too early.
            while len(_files) > _tmp_len_files:
                buf = {str(f_type): None for f_type in self._FILE_TYPE.keys()}
                tmp_files = copy.copy(_files)
                f = tmp_files[0][:-4]

                # File naming patter. Based off specified naming conventions
                pattern = f"^(\w+_{f})$|^({f})$"

                date = re.match(self._DATE_FORMAT, f)
                if date:

                    # Loop over remaining files and find ones with the same name
                    for i, file in enumerate(tmp_files):
                        # Create a pattern that is the file name omitting the file extension
                        patch_pattern = re.match(pattern, file[:-4])

                        if patch_pattern:
                            file_type = self._FILE_TYPE.get(file[-4:].lower())

                            if file_type is not None:
                                buf[file[-4:].lower()] = os.path.join(_relative_path, file)
                                files.remove(file)
                            if None not in buf.values():
                                # buf contains all associated files
                                return buf
                else:
                    del _files[0]
                # Did not find any Similar files
                if len(files) == len(tmp_files):
                    return None
                if None not in buf.values():
                    return buf

        patch_data = []

        relative_path = "".join(root.split(self._abs_path))[1:]
        while len(files) > 3:
            # Offset for 3 associated files being removed.
            tmp_len_files = len(files) - 3
            patch_files = _get_associated_files(files, tmp_len_files, relative_path)
            if patch_files is not None and patch_files not in patch_data:
                patch_data.append(patch_files)
            self.f.append(len(files))

        return None if not patch_data else patch_data

    def __getitem__(self, idx):
        """getitem method.

        Args:
            idx (int or torch.tensor): index of sample in the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self._abs_path, self._df.iloc[idx, 0])
        X = io.imread(img_name)
        X = Image.fromarray(X)

        if self._transform is not None:
            X = self._transform(X)

        # Dont think we need these files...
        # class_count_path = os.path.join(self._abs_path, self._df.iloc[idx, 2])
        # class_count = self._read_class_counts(class_count_path)

        landmarks_path = os.path.join(self._abs_path, self._df.iloc[idx, 1])
        y = self._read_landmarks(landmarks_path)

        if self._y_transform is not None:
            y = self._y_transform(y)

        return X, y


class WindowedImages(Dataset):
    def __init__(self, imgs_dataset, width, height):
        self._imgs_dataset = imgs_dataset
        self._w = width
        self._h = height
        self._idx = 0
        self._n_windowed_labels = 0
        self._windowed_data = None

    def _create_window(self, batch):
        # Batch: (image, dict of labels)
        data, labels = batch
        windowed_imgs, windowed_labels = [], []
        for label, center_pt in labels.items():
            # Skip empty label
            if not center_pt:
                continue
            for x, y in center_pt:
                windowed_img = data[:, x - self._w:x + self._w, y - self._h:y + self._h]
                windowed_imgs.append(windowed_img)
                windowed_labels.append(label)
        return windowed_imgs, windowed_labels

    def __getitem__(self, idx):
        """getitem method.

        Args:
            idx (int or torch.tensor): index of sample in the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch = self._imgs_dataset[idx]
        self._windowed_data = self._create_window(batch)


