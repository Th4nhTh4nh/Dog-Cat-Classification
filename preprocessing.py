from matplotlib import pyplot
from matplotlib.image import imread
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from os import makedirs
from shutil import copyfile
from random import seed
from random import random


folder = "train/"


def DogImgs(folder):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        filename = folder + "dog." + str(i) + ".jpg"
        image = imread(filename)
        pyplot.imshow(image)
    pyplot.show()


def CatImgs(folder):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        filename = folder + "cat." + str(i) + ".jpg"
        image = imread(filename)
        pyplot.imshow(image)
    pyplot.show()


# You can skip this function if your computer has less than 16GB of RAM
def ResizeImgs(folder):
    photos, labels = list(), list()
    for file in listdir(folder):
        output = 0.0
        if file.startswith("dog"):
            output = 1.0
        photo = load_img(folder + file, target_size=(200, 200))
        photo = img_to_array(photo)
        photos.append(photo)
        labels.append(output)
    photos = asarray(photos)
    labels = asarray(labels)
    print(photos.shape, labels.shape)
    save("dogs_vs_cats_photos.npy", photos)
    save("dogs_vs_cats_labels.npy", labels)


def RestructureDirectory():
    dataset_home = "dataset_dogs_vs_cats/"
    subdirs = ["train/", "test/"]
    for subdir in subdirs:
        labeldirs = ["dogs/", "cats/"]
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)
    seed(1)
    val_ratio = 0.25
    src_directory = "train/"
    for file in listdir(src_directory):
        src = src_directory + "/" + file
        dst_dir = "train/"
        if random() < val_ratio:
            dst_dir = "test/"
        if file.startswith("cat"):
            dst = dataset_home + dst_dir + "cats/" + file
            copyfile(src, dst)
        elif file.startswith("dog"):
            dst = dataset_home + dst_dir + "dogs/" + file
            copyfile(src, dst)


# DogImgs(folder)
# CatImgs(folder)
RestructureDirectory()
