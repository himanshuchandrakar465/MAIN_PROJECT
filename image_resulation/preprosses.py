import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


image_paths = Path(
    r"C:\Users\ASUS\.cache\kagglehub\datasets\joe1995\div2k-dataset\versions\1\DIV2K_train_HR\DIV2K_train_HR"
)

image_path = image_paths / "0001.png"

items = list(image_paths.iterdir())

number_of_image = 10


def image_embaddings(
    image_path=image_path,
):  # -------> this function canvert images to numbers if you give the path

    image = cv2.imread(image_path)

    # print(type(image))
    # print(image)
    return image


def show(
    image,
):  # --------> this function show the image if you give the embaddings / numbers of the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.show()


def shape_of_image(
    image_path,
):  # --------> this function give the shape of the matrix / numbers 2d list

    image_shape = image_embaddings(image_path).shape

    return image_shape


def random_value(
    image_path,
):  # ------>this function give the the random value between the image coodinates if you give the image path
    x = shape_of_image(image_path)[0]
    y = shape_of_image(image_path)[1]
    a = np.random.randint(0, x + 1)
    b = np.random.randint(0, y + 1)

    return a, b


def crop_image(
    image_path,
):  # ------> this function give the croped image if you give the image path
    image = image_embaddings(image_path)

    h, w = image.shape[:2]
    width, height = random_value(image_path)

    x = (w - width) // 2
    y = (h - height) // 2

    return image[y : y + height, x : x + width]


def image_genrater(image_link):  # ------> this function genrate images from one images
    list_of_different_images_from_one_images = []
    for i in range(0, number_of_image):
        list_of_different_images_from_one_images.append(crop_image(image_link))
    return list_of_different_images_from_one_images


if __name__ == "__main__":
    # show(image=image_embaddings()) #-----> this is how you can see the image
    for i in items[0:2]:
        for j in image_genrater(i):
            show(j)
