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


def image_embaddings(image_path):
    return cv2.imread(str(image_path))


def show(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def shape_of_image(image):
    return image.shape


def random_value(image):
    h, w = image.shape[:2]

    height = np.random.randint(1, h + 1)
    width = np.random.randint(1, w + 1)

    return width, height


def crop_image(image):
    h, w = image.shape[:2]

    crop_height = np.random.randint(64, h + 1)
    crop_width = np.random.randint(64, w + 1)

    x = np.random.randint(0, w - crop_width + 1)
    y = np.random.randint(0, h - crop_height + 1)

    return image[y : y + crop_height, x : x + crop_width]


def image_genrater(image):
    return [crop_image(image) for _ in range(number_of_image)]


def genrate_more_rotated_image(image):
    final_image_list = []

    for img in image_genrater(image):
        final_image_list.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

        final_image_list.append(cv2.rotate(img, cv2.ROTATE_180))

        final_image_list.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    return final_image_list


def saving_the_genrated_images(image_link):

    x_folder = Path(
        r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_x"
    )

    y_folder = Path(
        r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_y"
    )

    x_folder.mkdir(parents=True, exist_ok=True)
    y_folder.mkdir(parents=True, exist_ok=True)

    image = image_embaddings(image_link)

    generated_images = genrate_more_rotated_image(image)

    existing_files = list(x_folder.glob("*.jpg"))

    start_number = len(existing_files)

    for j, generated_image in enumerate(generated_images):
        number = start_number + j

        filename = f"{number}.jpg"

        cv2.imwrite(
            str(x_folder / filename),
            generated_image,
        )

        cv2.imwrite(
            str(y_folder / filename),
            generated_image,
        )


def delete_images():
    folders = [
        Path(
            r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_x"
        ),
        Path(
            r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_y"
        ),
    ]

    for folder in folders:
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()


def make_xtrain_rough(xtrain_path):

    xtrain_path = Path(xtrain_path)

    for image_path in xtrain_path.iterdir():
        if image_path.is_file():
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            # Make image rough
            rough_image = cv2.GaussianBlur(image, (5, 5), 0)

            # Add noise
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)

            rough_image = cv2.add(rough_image, noise)

            # Same path + same filename
            cv2.imwrite(str(image_path), rough_image)


if __name__ == "__main__":
    delete_images()

    for image_path in items[:3]:
        saving_the_genrated_images(image_path)

    make_xtrain_rough(
        r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_x"
    )
