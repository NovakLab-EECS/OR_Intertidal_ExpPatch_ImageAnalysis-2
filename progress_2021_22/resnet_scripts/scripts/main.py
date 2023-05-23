import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import logging
import torch
import torch.nn.functional as F

from dataloader.dataset import PatchPicsDataset
from os.path import dirname, abspath, join
from trainer.train import Trainer
from torch.utils.data.dataset import Dataset

from PIL import Image

logging.basicConfig(
    format="%(asctime)s [%(name)s: %(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("main.py")

lr = 0.01
batch_size = 1


def device_selector():
    """Select what device to run training on.
    Uses cuda if available.

    Returns: (torch.device) Device the training is going to be run on.
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info(
            "CUDA version: [ %s ] | CUDA DEVICE: [ %s ]",
            torch.version.cuda,
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )
    return torch.device("cuda" if use_cuda else "cpu")


def windowed_collate_fn(batch):
    w, h = 256, 256
    # Batch: (image, dict of labels)
    # Get num of labels in a batch to preallocate tensor space
    n_labels = sum([len(k) for k in batch[0][1].values()])
    windowed_imgs, windowed_labels = torch.zeros((n_labels, 3, 2 * w, 2 * h)), torch.zeros(n_labels)

    i = 0
    for data, labels in batch:
        C, X_max, Y_max = data.shape
        for label, center_pt in labels.items():
            # Skip empty label
            if not center_pt:
                continue
            for y, x in center_pt:  # IDK what is wrong with the points... if we do x,y it is out of bounds
                assert x <= X_max
                assert y <= Y_max

                # Padding bounds need to account for edge cases.
                w_lower_pad, w_upper_pad = max(0, x - w), min(X_max, x + w)
                h_lower_pad, h_upper_pad = max(0, y - h), min(Y_max, y + h)

                windowed_img = data[:, w_lower_pad:w_upper_pad, h_lower_pad:h_upper_pad]
                # im1 = img_transform(windowed_img)
                # im1.show()
                windowed_c, windowed_w, windowed_h = windowed_img.shape
                pad = (((2 * h - windowed_h) // 2), ((2 * h - windowed_h + 1) // 2), ((2 * w - windowed_w) // 2),
                       ((2 * w - windowed_w + 1) // 2))
                padded_img = F.pad(
                    input=windowed_img,
                    pad=pad,
                    mode='constant',
                )
                # im = img_transform(padded_img)
                # im.show()

                # Verify that shape of the padded image is the expected shape.
                assert padded_img.shape == (3, 2 * w, 2 * h), (padded_img.shape, (3, 2 * w, 2 * h), i)
                windowed_imgs[i] = padded_img
                windowed_labels[i] = label
                i += 1

    return windowed_imgs, windowed_labels


def main():
    device = device_selector()

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    # Get absolute path to `ExpPatch-Pics` directory
    abs_path = dirname(dirname(abspath(__file__)))
    patch_path = join(abs_path, "ExpPatch-Pics")

    train_set = PatchPicsDataset(dataset_path=patch_path, transform=transform)
    # test_set = PatchPicsDataset(dataset_path="wherever test data is")

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=windowed_collate_fn,
    )

    #t = windowed_collate_fn(train_set[0])
    # print(type(train_set[0][0]), type(train_set[0][1]))
    # print(f"shape trainset[0]: {train_set[0][0].shape}")
    # print(f'train set[0]:\n{train_set[0][0]}\n-----------------------------------------')
    # print(f'train set[1]:\n{train_set[0][1]}\n-----------------------------------------')


    # testloader = torch.utils.data.DataLoader(
    #     test_set, batch_size=batch_size, shuffle=True, num_workers=0
    # )

    model = models.resnet50(pretrained=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    trainer = Trainer(
        data_loader=trainloader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    trainer.fit_model(1)


if __name__ == "__main__":
    main()
