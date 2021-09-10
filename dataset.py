import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from dataset_generator import get_labels
from torchvision import transforms
import cv2

IMAGE_SIZE = 224


class GarbageData(Dataset):
    def __init__(self, csv_path, json_path, train_flag=True):
        super().__init__()

        # Load the cvs file
        self.data_info = pd.read_csv(csv_path, header=None)
        self.train_flag = train_flag
        self.label_dict = get_labels(json_path)

        self.img_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        assert self.img_arr.shape == self.label_arr.shape
        self.dataset_size = self.img_arr.shape[0]

        self.train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

        self.valid_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.img_arr[index]
        img = Image.open(img_path)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.valid_tf(img)

        # img = self.transform(img)

        label_idx = self.label_arr[index]
        assert label_idx in self.label_dict.keys()

        return img, label_idx, self.label_dict[label_idx]

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    writer = SummaryWriter('logs')
    csv_path = './dataset.csv'
    json_path = './labels.json'

    # Dataset test
    test_idx = 1880
    dataset = GarbageData(csv_path, json_path, train_flag=True)
    print("dataset size: {}".format(len(dataset)))
    img, label_idx, label = dataset(test_idx)
    title = "image {} ".format(test_idx) + label
    writer.add_image(title, img)

    # Dataloader test
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True)

    for idx, data in enumerate(dataloader):
        img_tensor, label_idx, label = data

        if idx % 50 == 0:
            print(img_tensor.shape)
            print(label_idx.shape)
