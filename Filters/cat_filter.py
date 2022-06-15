import numpy as np
import matplotlib.pyplot as plt
import cv2


# TODO: Change this to the path to the project on your hard drive
PROJECT_PATH = "D:/faks/Deep learning/Project/dl_project"


class CatFilter:
    """
    Parent class for all the filter classes that place different filters over cat.
    """
    def __init__(self):
        self.filter_dir = PROJECT_PATH + "/Filters"
        pass

    def place_filter(self, img, keypoints):
        return img

    def compute_iou(self, img, y_true, y_pred):
        """
        Function that computes the intersection over union for filters placed on the image based on the predicted
        and true keypoints.
        :param img: cv2 image - image, over which the filters are placed
        :param y_true: list - true keypoints
        :param y_pred: list - predicted keypoints
        :return: float - computed IoU
        """
        # place the filters over empty images based on predicted and true keypoints and obtain binary masks
        mask_img = np.zeros(img.shape, dtype=np.uint8)
        pred_mask = np.sum(self.place_filter(mask_img, y_pred), axis=2)
        true_mask = np.sum(self.place_filter(mask_img, y_true), axis=2)
        pred_mask = pred_mask > 0
        true_mask = true_mask > 0

        intersection = np.sum(np.logical_and(pred_mask, true_mask))
        union = np.sum(np.logical_or(pred_mask, true_mask))

        if union != 0:
            return intersection / union

        return 0
