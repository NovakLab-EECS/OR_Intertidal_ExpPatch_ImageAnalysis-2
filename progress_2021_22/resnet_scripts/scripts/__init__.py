"""Scripts module."""

#from OR_Intertidal_ExpPatch_ImageAnalysis.scripts.dataloader import dataloader
from scripts.trainer import Trainer
from scripts.dataloader import PatchPicsDataset, PatchDataset

__all__ = (
    "PatchDataset",
    "PatchPicsDataset",
    "Trainer",
)
