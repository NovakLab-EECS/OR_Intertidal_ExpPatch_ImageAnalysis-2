from os.path import dirname, abspath, join
from torch.utils.data import DataLoader

from dataset import (
    PatchPicsDataset,
)


def test_patch_dataset():
    patch_path = "ExpPatch-Pics"
    batch_size = 10

    # Get absolute path to `ExpPatch-Pics` directory
    abs_path = dirname(dirname(dirname(abspath(__file__))))
    patch_path = join(abs_path, patch_path)

    patch_pic_dataset = PatchPicsDataset(dataset_path=patch_path)
    # patch_pic_dataset.open_image()
    # patch_pic_dataset.patches_dict_to_file()
    test_return_value = patch_pic_dataset[0]

    dataloader = DataLoader(
        dataset=patch_pic_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    # print(f'Returned value: {test_return_value}')


if __name__ == "__main__":
    test_patch_dataset()
