import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from imports import load_synset_mapping
import torch
from utils import RandAugment
#########################
class ImageNetValDataset(Dataset):
    def __init__(self, img_dir, label_file, synset_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Load synset mapping (WordNet ID → Class Index)
        self.synset_mapping = load_synset_mapping(synset_file)

        # Load the CSV with ground truth labels
        self.labels = pd.read_csv(label_file, header=0, usecols=[0, 1], names=["filename", "label"], delimiter=",")

        # Ensure filenames are properly formatted
        self.labels["filename"] = self.labels["filename"].astype(str).str.strip()
        self.labels["label"] = self.labels["label"].astype(str).str.strip()

        # Remove bounding box information (only keep first word of label)
        self.labels["label"] = self.labels["label"].apply(lambda x: x.split(" ")[0])

        # Append `.JPEG` if missing in filenames
        self.labels["filename"] = self.labels["filename"].apply(lambda x: x + ".JPEG" if not x.endswith(".JPEG") else x)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        wnid_label = self.labels.iloc[idx, 1]  # WordNet ID

        # Convert WordNet ID to class index
        if wnid_label in self.synset_mapping:
            label = self.synset_mapping[wnid_label]
        else:
            raise ValueError(f"❌ Label {wnid_label} not found in synset mapping!")

        img_path = os.path.join(self.img_dir, img_name)

        # Debugging: Print full path before loading
        # print(f"🔍 Checking file: {img_path}")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            raise ValueError("❌ No transform function provided!")

            # Convert label to a PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor

        return image, label


def imageNET(datasetdir, size=256, batchsize=64):

    val_img_dir = os.path.join(datasetdir, 'ImageNet', 'data', 'val')
    label_file = os.path.join(datasetdir, 'ImageNet', 'data', 'LOC_val_solution.csv')
    synset_file = os.path.join(datasetdir, 'ImageNet', 'data', 'LOC_synset_mapping.txt')

    transf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure synset file is passed
    # print(f'-I({__file__}): Loading ImageNet dataset from {val_img_dir}')

    test_dataset = ImageNetValDataset(val_img_dir, label_file, synset_file, transform=transf)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batchsize,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=torch.cuda.is_available())

    print(f'-I({__file__}): ImageNet loaded')

    return test_loader


