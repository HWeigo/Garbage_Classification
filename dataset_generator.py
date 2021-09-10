import json
import os
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


# --- END ---

# --- Rename image name by {label}_{item}.jpg ---
def get_labels(json_path: str):
    """
    Read .json file
    :param json_path: path to json file
    :return: data: label dictionary
    """
    with open(json_path) as file:
        dict = json.load(file)
        return dict


def rename_images(dataset_path: str):
    """
    Rename image's name
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


if __name__ == '__main__':
    dataset_path = "./data"
    json_path = "labels.json"

    # get labels
    # labels_dict = get_labels(json_path)

    # rename image
    # rename_images(dataset_path)
