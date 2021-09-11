import csv
import json
import os
import random
from sklearn.model_selection import train_test_split

import pandas as pd


def rename_directory():
    """
    Rename directory name
    """

    root_dir = '../Datasets/MyDataset/data/'
    labels = os.listdir(root_dir)
    new_labels = [str(int(l) - 6) for l in labels]
    print(labels)
    print(new_labels)

    for i in range(len(labels)):
        ori_dir_name = os.path.join(root_dir, labels[i])
        print(ori_dir_name)
        new_dir_name = os.path.join(root_dir, new_labels[i])
        print(new_dir_name)
        try:
            os.rename(ori_dir_name, new_dir_name)
        except OSError as err:
            print("[ERROR ]OS error: {0}. Fail to rename directory.".format(err))


def get_labels(json_path: str):
    """
    Read .json file

    :param json_path: path to json file
    :return: data: label dictionary
    """

    with open(json_path) as file:
        dict = json.load(file)
        dict = {int(k): v for k, v in dict.items()}
        return dict


def rename_images(dataset_path: str):
    """
    Rename image's name by {label}_{item}.jpg

    :param dataset_path: path to dataset
    """

    labels = os.listdir(dataset_path)
    print(labels)

    for i in range(len(labels)):
        subdir = os.path.join(dataset_path, labels[i])
        ori_imgs_name = os.listdir(subdir)
        imgs_size = len(ori_imgs_name)
        print("class " + labels[i] + " have {} images".format(imgs_size))
        for j in range(imgs_size):
            ori_file = os.path.join(subdir, ori_imgs_name[j])
            new_name = "{}_{}.jpg".format(int(labels[i]), j)
            new_file = os.path.join(subdir, new_name)
            try:
                os.rename(ori_file, new_file)
            except:
                print("[ERROR] fail to rename image for {} image in class {}".format(j, int(labels[i])))


def csv_generator(csv_name, dataset_path):
    """
    Generate csv file for pytorch dataloader

    :param csv_name: name of csv file
    :param dataset_path: path to dataset
    :return: length of the dataset
    """

    subdir_name = os.listdir(dataset_path)
    sum_img_paths = []
    sum_labels = []
    for i in subdir_name:
        subdir_path = os.path.join(dataset_path, i)
        print(subdir_path)
        img_names = os.listdir(subdir_path)
        img_paths = [subdir_path + '/' + n for n in img_names]
        labels = [int(i)] * len(img_names)
        assert len(img_paths) == len(labels)
        sum_img_paths += img_paths
        sum_labels += labels

    rows = zip(sum_img_paths, sum_labels)
    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    return len(sum_labels)


def split_dataset(dataset_path, train_path, valid_path, test_path, percent):
    """
    Split the original dataset into training, validation and test dataset

    :param dataset_path: path to original dataset csv file
    :param train_path: path to training dataset csv file
    :param valid_path: path to validation dataset csv file
    :param test_path: path to test dataset csv file
    :return: None
    """

    assert len(percent) == 3
    train_percent, valid_percent, test_percent = percent

    df = pd.read_csv(dataset_path)
    trainval, test = train_test_split(df, test_size=test_percent, random_state=7)
    train, valid = train_test_split(trainval, test_size=valid_percent, random_state=12)

    print("train: {}, valid: {}, test: {}".format(train.shape[0], valid.shape[0], test.shape[0]))

    train = train.sort_index(axis=0)
    valid = valid.sort_index(axis=0)
    test = test.sort_index(axis=0)

    train.to_csv("train.csv", header=False, index=False)
    valid.to_csv("valid.csv", header=False, index=False)
    test.to_csv("test.csv", header=False, index=False)


if __name__ == '__main__':
    dataset_path = "./data"
    json_path = "labels.json"

    # Get labels
    # labels_dict = get_labels(json_path)

    # Rename image
    # rename_images(dataset_path)

    # Generate csv file
    # dataset_size = csv_generator("dataset.csv", dataset_path)
    # print("Total samples in the dataset: {}".format(dataset_size))

    # Split dataset
    split_dataset("dataset.csv", "train.csv", "valid.csv", "test.csv", [0.75, 0.2, 0.05])
