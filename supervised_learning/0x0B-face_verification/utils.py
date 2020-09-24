#!/usr/bin/env python3
"""
File with utils methods:
    load_images
    load_csv
    save_images
"""

import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt


def load_images(images_path, as_array=True):
    """ Load all images from a directory
    Args:
        images_path: is the path to a directory from which to load images
        as_array: is a boolean indicating whether the images should be
            loaded as one numpy.ndarray
    Returns:
        images: is either a list/numpy.ndarray of all images
        filenames: is a list of the filenames associated with each image
            in images
    """
    images = []
    filenames = []
    # Sorted by name the images in the path
    list_dir = os.listdir(images_path)

    for path in sorted(list_dir):
        path_image = (images_path + "/" + path)
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        filenames.append(path)

    # Create a numpy array if as_array is true
    if as_array:
        images = np.stack(images)

    return (images, filenames)


def load_csv(csv_path, params={}):
    """  Loads the contents of a csv file as a list of lists
    Args:
        csv_path: Is the path to the csv to load
        params: are the parameters to load the csv with
    Returns:
        a list of lists representing the contents found in csv_path
    """
    csv_values = []
    with open(csv_path, encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, params)
        # Reads all lines of file
        for row in spamreader:
            csv_values.append(row)
    return csv_values


def save_images(path, images, filenames):
    """ Saves images to a specific path
    Args:
        path: is the path to the directory in which the images should be saved
        images: is a list/numpy.ndarray of images to save
        filenames: is a list of filenames of the images to save
    Returns:
        True on success and False on failure
    """

    # Check if path exists, if not exists, create directory
    if os.path.exists(path):
        # os.makedirs(path)
        for img, name in zip(images, filenames):
            # convert the image into a RGB format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./{}/{}".format(path, name), image)
        return True
    else:
        return False


def generate_triplets(images, filenames, triplet_names):
    """ Generates triplets:
    Args:
        images: is a numpy.ndarray of shape (n, h, w, 3) containing the various
            images in the dataset
        filenames:  is a list of length n containing the corresponding
            filenames for images
    Returns:
        a list [A, P, N]
            A - is a numpy.ndarray of shape (m, h, w, 3) containing the anchor
                images for all m triplets
            P - is a numpy.ndarray of shape (m, h, w, 3) containing the
                positive images for all m triplets
            N - is a numpy.ndarray of shape (m, h, w, 3) containing the
                negative images for all m triplets
    """
    names = []

    # Storage the name of the files without .jpg
    for i in filenames:
        my_name = i.split('.')[0]
        names.append(i.split('.')[0])

    index = []
    a = []
    p = []
    n = []
    # Reeplace values with differents characters
    for triplet_name in triplet_names:
        for name in triplet_name:
            if name not in names:
                # Reeplace special characters as í, é, ñ
                new_value = name.encode('utf-8').decode('utf-8')
                new_value = new_value.replace('eÌ', 'é')
                new_value = new_value.replace('iÌ', chr(105) + chr(769))
                new_value = new_value.replace('nÌƒ', chr(110)+chr(771))
                new_value = new_value.replace('\x81', '')
                new_value = new_value.replace('\x81', '')

                # Find the name of the triplet in the names array and replace
                # it
                triplet_name[triplet_name.index(name)] = new_value

    # Create the list with indices of the values
    for triplet_name in triplet_names:
        a.append(names.index(triplet_name[0]))
        p.append(names.index(triplet_name[1]))
        n.append(names.index(triplet_name[2]))

    # Create arrays with the images
    A = images[a]
    P = images[p]
    N = images[n]

    return [A, P, N]
