import numpy as np
import pandas as pd

from keras.utils import Sequence
from keras_preprocessing.image import load_img, img_to_array


class DataGenerator(Sequence):
    """
    Data generator for cats image data set. This generator can be used for train, test or val set (each set should be
    in separate folder). The structure of the folder representing the data should be as follows - the images need to
    be in subfolder "images" and annotations should be provided in file "labels.csv", where each row is represented
    by the name of the image and x and y coordinate of each of 9 key points
    """
    def __init__(self, data_path, batch_size, shuffle):
        """
        :param data_path: path to the folder, where the images and annotations are stored
        :param batch_size: int, the size of the batch
        :param shuffle: boolean, whether data is shuffled after the each epoch
        """
        self.data_path = data_path
        self.df = pd.read_csv(data_path + "/labels.csv").values
        # Shuffle the data for first epoch if we are shuffling
        if shuffle:
            np.random.shuffle(self.df)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __getitem__(self, index):
        batch = self.df[self.batch_size*index:self.batch_size*(index+1), :]

        # The filenames are stored in the first column of the data
        X = np.asarray([self.load_image(filename) for filename in batch[:, 0]], dtype="uint8")
        # The remaining columns represent data outputs
        y = batch[:, 1:].astype("int32")

        return X, y

    def load_image(self, filename):
        """
        Helper function that loads the image as numpy array
        :param filename: name of the file in which the image is stored
        :return:
        """
        img = load_img("/".join([self.data_path, "images", filename]))
        img = img_to_array(img)

        return img

    def __len__(self):
        size = self.df.shape[0] // self.batch_size

        # In case the last batch is smaller
        if self.df.shape[0] % self.batch_size != 0:
            size += 1

        return size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.df)
