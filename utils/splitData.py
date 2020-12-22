import os
import fnmatch
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def find_image_names(image_dir, extension="png"):
    """
    Find image names for the dataset
    :param image_dir: Directory where images are ("data/COVID")
    :param extension: Image extenion (png, jpg, etc)
    :return: list containing the name of all images in the directory
    """
    cwd = Path.cwd().parent
    image_dir = os.path.join(cwd, image_dir)
    image_extension = "*." + extension
    image_names = fnmatch.filter(os.listdir(image_dir), image_extension)
    image_names = [os.path.join(image_dir, name) for name in image_names]
    return image_names


def split_dataset(image_names, label_val=0):
    """
    Split images in train, test, val
    :param label_val: Encoding value of label (e.g., 1 for COVID)
    :param image_names:
    :return:
    """
    labels = [label_val] * len(image_names)
    X_train, X_test, Y_train, Y_test = train_test_split(
        image_names, labels, test_size=0.25, random_state=14
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.15, random_state=25
    )
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def save(data, name):
    X, y = data
    df = pd.DataFrame(data={"X": X, "y": y})
    df.to_csv(name)


def main(root_dir):
    """

    :param root_dir: Dir with COVID and non-COVID imgs.
    :return:
    """
    # get filenames of images
    covid_images_names = find_image_names(os.path.join(root_dir, "COVID"))
    non_covid_images_names = find_image_names(os.path.join(root_dir, "non-COVID"))
    # split images in train, val, test
    X_train_covid, X_val_covid, X_test_covid, Y_train_covid, Y_val_covid, Y_test_covid = split_dataset(
        covid_images_names, label_val=1
    )
    X_train_non_covid, X_val_non_covid, X_test_non_covid, \
    Y_train_non_covid, Y_val_non_covid, Y_test_non_covid = split_dataset(
        non_covid_images_names, label_val=0
    )
    # Merge covid and non-covid and shuffle
    X_train, Y_train = X_train_covid + X_train_non_covid, Y_train_covid + Y_train_non_covid
    X_val, Y_val = X_val_covid + X_val_non_covid, Y_val_covid + Y_val_non_covid
    X_test, Y_test = X_test_covid + X_test_non_covid, Y_test_covid + Y_test_non_covid
    save(shuffle(X_train, Y_train), name="train_set.csv")
    save(shuffle(X_val, Y_val), name="val_set.csv")
    save(shuffle(X_test, Y_test), name="test_set.csv")


if __name__ == '__main__':
    main("data")
