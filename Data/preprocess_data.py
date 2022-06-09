import cv2
import os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

# The width and height of the images after preprocessing
TARGET_WIDTH = 224
TARGET_HEIGHT = 224


class DataPreprocessor:
    def __init__(self, path):
        self.path = path
        self.train_set = []
        self.val_set = []
        self.test_set = []

    def split_data(self, train_size=0.8, val_size=0.1, seed=5):
        """
        Splits the images into the train, validation and test set. Randomly splits the images inside each of the
        subfolders CAT_00 to CAT_06. The function stores the filenames of the images that belong to each split in
        lists train_set, val_set and test_set
        :param train_size: float between 0 and 1, the proportion of images that go to train set
        :param val_size: float between 0 and 1, the proportion of images that go to test set
        :param seed: int, the seed of random generator
        :return:
        """
        np.random.seed(seed)

        dirs = os.listdir(self.path)

        print("Splitting the data into train, test and val set ...")

        for dir in dirs:
            # Get the images in the directory and random shuffle them
            imgs = np.array([f"{dir}/{file}" for file in os.listdir("/".join([self.path, dir])) if file[-4:] == ".jpg"])
            np.random.shuffle(imgs)

            # Get the length of train and val set and put first train_length images into train set, next val_length
            # images into val set and remaining images into test set
            train_length = int(np.round(train_size*len(imgs)))
            val_length = int(np.round(val_size*len(imgs)))
            self.train_set += imgs[:train_length].tolist()
            self.val_set += imgs[train_length:train_length+val_length].tolist()
            self.test_set += imgs[train_length+val_length:].tolist()

        print("Done!")

    def process_split(self, split):
        """
        Function that processes and stores the data for given split.
        :param split: split that will be processed, either "train", "test" or "val"
        :return:
        """
        # If the split data is already stored delete it, otherwise create the split folder
        if os.path.exists(split):
            os.removedirs(split)
        else:
            os.makedirs(split)
            os.makedirs(split + "/images")

        if split == "train":
            split_data = self.train_set
        elif split == "test":
            split_data = self.test_set
        elif split == "val":
            split_data = self.val_set
        else:
            raise ValueError("Illegal data split", split)

        print(f"Processing {split} data ...")

        df = []
        for file in tqdm(split_data):
            img, annot = self.resize_image(file)
            img_name = "_".join(file.split("/"))
            # Store the resized image
            cv2.imwrite("/".join([split, "images", img_name]), img)
            # Add the annotations together with the image name to data frame
            df.append([img_name] + annot)

        column_names = ["image",
                        "left_eye_x", "left_eye_y",
                        "right_eye_x", "right_eye_y",
                        "mouth_x", "mouth_y",
                        "left_ear1_x", "left_ear1_y",
                        "left_ear2_x", "left_ear2_y",
                        "left_ear3_x", "left_ear3_y",
                        "right_ear1_x", "right_ear1_y",
                        "right_ear2_x", "right_ear2_y",
                        "right_ear3_x", "right_ear3_y"]

        df = pd.DataFrame(data=df, columns=column_names)
        df.to_csv(split + "/labels.csv", index=False)

        print("Done!")

    def resize_image(self, filename):
        """
        Helper function that returns the resized image and the annotations that are also resized.
        :param filename: Name of the subfolder in cats directory and name of the image in that directory
        :return: cv2 image, list -> the resized image and list of 18 coordinates that represent the resized points
        """
        img = cv2.imread("/".join([self.path, filename]))
        annotations = open("/".join([self.path, filename]) + ".cat", "r")
        # Read the coordinates and skip the first value as it just represents the number of points
        points = annotations.read().strip().split(" ")[1:]
        # Convert the read strings into integers
        points = list(map(lambda el: int(el), points))

        # Compute the ratio with which the height and width will be multiplied
        w_ratio = TARGET_WIDTH / img.shape[1]
        h_ratio = TARGET_HEIGHT / img.shape[0]

        # Resize the image
        resized_img = cv2.resize(img, dsize=(TARGET_WIDTH, TARGET_HEIGHT))
        # Compute the coordinates of the points after resizing
        resized_pts = []
        for x, y in zip(points[::2], points[1::2]):
            # Mulitply the points by the ratio and ruond them to integer value
            resized_x = int(np.round(x * w_ratio))
            resized_pts.append(resized_x)
            resized_y = int(np.round(y * h_ratio))
            resized_pts.append(resized_y)

        return resized_img, resized_pts


if __name__=="__main__":
    preprocessor = DataPreprocessor("../cats")
    preprocessor.split_data(train_size=0.8, val_size=0.1)
    preprocessor.process_split("train")
    preprocessor.process_split("val")
    preprocessor.process_split("test")
