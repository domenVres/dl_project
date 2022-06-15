import cv2
import pandas as pd
import matplotlib.pyplot as plt
from Filters.cat_filter import CatFilter

import warnings


class HatFilter(CatFilter):
    """
    Filter that puts the glasses over the cat's face. The widht of the hat is determined as the distance between the
    leftmost point of the left ear and the rightmost point of the right ear. The bottom point of the hat is determinned
    as the middle point betwen left ear's and left eye's height. The angle of the hat is also adjusted according to the
    tilt of cat's head.
    """
    def __init__(self):
        super().__init__()
        self.filter = cv2.imread("/".join([self.filter_dir, "hat.png"]), -1)
        self.h, self.w = self.filter.shape[:2]

    def place_filter(self, img, keypoints):
        """
        Function that places the hat filter over the cat image
        :param img: cv2 image - the image over which the filter will be placed
        :param keypoints: iterable object - the keypoints in the format that is produced by the model
        :return: cv2 image - the image with hat on top of the cat's face
        """
        # Get the coordinates of left and right eye
        left_ear = keypoints[6:8]
        right_ear = keypoints[16:]
        left_eye = keypoints[:2]

        # Resize the glasses to the width proportional to the distance between eyes while preserving the aspect ratio
        w_new = right_ear[0] - left_ear[0]
        r = w_new / self.w
        h_new = int(r * self.h)
        if w_new > 0 and h_new > 0:
            hat = cv2.resize(self.filter, (w_new, h_new))
        else:
            warnings.warn("Right ear is located left of the left ear. Placing of the filter unsuccessful.")
            return img

        # Use the transparency as well so the filter can be overlayed
        img_filtered = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Angle between eears (if the head is tilted ears are not on a horizontal line)
        angle_coefficient = (right_ear[1] - left_ear[1]) / (right_ear[0] - left_ear[0])

        # Overlap the filter over the image
        h_img, w_img = img_filtered.shape[:2]
        left_most = left_ear[0]
        bottom_point = (left_eye[1] + left_ear[1]) // 2
        for i in range(h_new):
            for j in range(w_new):
                if hat[i, j][3] != 0:
                    x = left_most + j
                    y = bottom_point - h_new + i + int(j * angle_coefficient)
                    if 0 <= x < w_img and 0 <= y < h_img:
                        img_filtered[y, x] = hat[i, j]

        # Revert back to BGR
        img_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_BGRA2BGR)

        return img_filtered


if __name__=="__main__":
    df = pd.read_csv("../Data/train/labels.csv")
    x = df.values[0]
    img = cv2.imread("../Data/train/images/" + x[0])
    points = [int(pt) for pt in x[1:]]
    h_filter = HatFilter()
    filtered_img = h_filter.place_filter(img, points)

    plt.subplot(1, 2, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Before")

    plt.subplot(1, 2, 2)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    plt.imshow(filtered_img)
    plt.title("After")

    plt.show()
