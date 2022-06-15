import warnings

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from Filters.cat_filter import CatFilter


class BeardFilter(CatFilter):
    """
    Filter that puts the glasses over the cat's face. The widht of the glasses is determined as 2 times the distance
    between the eyes. The angle of the glasses is also adjusted according to the tilt of cat's head. The filter was
    inspired by https://github.com/agrawal-rohit/opencv-facial-filters.
    """
    def __init__(self):
        super().__init__()
        self.filter = cv2.imread("/".join([self.filter_dir, "beard.png"]), -1)
        self.h, self.w = self.filter.shape[:2]

    def place_filter(self, img, keypoints):
        """
        Function that places the glasses filter over the cat image
        :param img: cv2 image - the image over which the filter will be placed
        :param keypoints: iterable object - the keypoints in the format that is produced by the model
        :return: cv2 image - the image with glasses over the cat's face
        """
        # Get the coordinates of left and right eye
        left_eye = keypoints[:2]
        right_eye = keypoints[2:4]
        mouth = keypoints[4:6]
        left_ear = keypoints[6:8]
        right_ear = keypoints[16:]

        # Resize the glasses to the width proportional to the distance between eyes while preserving the aspect ratio
        w_new = right_ear[0] - left_ear[0]
        r = w_new / self.w
        h_new = int(r * self.h)
        if w_new > 0 and h_new > 0:
            beard = cv2.resize(self.filter, (w_new, h_new))
        else:
            warnings.warn("Right ear is located left to the left ear. PLacing of the filter unsuccessful.")
            return img

        # Use the transparency as well so the filter can be overlayed
        img_filtered = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Angle between eyes (if the head is tilted eyes are not on a horizontal line)
        if right_eye[0] - left_eye[0] > 0:
            angle_coefficient = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
        else:
            angle_coefficient = 0

        # Overlap the filter over the image
        h_img, w_img = img_filtered.shape[:2]
        left_most = (left_eye[0] + right_eye[0] - w_new) // 2
        top_left = (left_eye[1] + mouth[1]) // 2
        for i in range(h_new):
            for j in range(w_new):
                if beard[i, j][3] != 0:
                    x = left_most + j
                    y = top_left + i + int(j * angle_coefficient)
                    if 0 <= x < w_img and 0 <= y < h_img:
                        img_filtered[y, x] = beard[i, j]

        # Revert back to BGR
        img_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_BGRA2BGR)

        return img_filtered


if __name__=="__main__":
    df = pd.read_csv("../Data/train/labels.csv")
    x = df.values[0]
    img = cv2.imread("../Data/train/images/" + x[0])
    points = [int(pt) for pt in x[1:]]
    b_filter = BeardFilter()
    filtered_img = b_filter.place_filter(img, points)

    plt.subplot(1, 2, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Before")

    plt.subplot(1, 2, 2)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    plt.imshow(filtered_img)
    plt.title("After")

    plt.show()
