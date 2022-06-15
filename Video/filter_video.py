import cv2
import time
import numpy as np
import argparse

from Filters.glasses import GlassesFilter
from Filters.hat import HatFilter
from Filters.beard import BeardFilter

from Models.mobilenet_v2 import MobileNetV2Tuned

# The height and width of the image that is provided as the input to the model
MODEL_DIM = 224


class VideoFilter:
    def __init__(self, glasses, beard, hat):
        self.filters = []
        self.filter_names = []
        if glasses:
            self.filters.append(GlassesFilter())
            self.filter_names.append("Glasses")
        if beard:
            self.filters.append(BeardFilter())
            self.filter_names.append("Beard")
        if hat:
            self.filters.append(HatFilter())
            self.filter_names.append("Hat")

        self.model = MobileNetV2Tuned()
        self.model.load_model("mobilenet_v2_mse.h5")
        self.window_name = ""

    def filter_video(self, path):
        self.window_name = path.split("/")[-1].split(".")[0]
        self.initialize_window()

        video = cv2.VideoCapture(path)

        # Variables for computing average time required for processing the frames
        model_time = 0
        filter_times = [0 for _ in self.filters]
        total_time = 0
        n_frames = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("Unable to read the frame, experiment stopped.")
                break

            h, w = frame.shape[:2]
            n_frames += 1

            # Start measuring the time (reading from the drive is excluded)
            total_start = time.time()

            # Resize the image for the correct input to the model and obtain the keypoints
            img = cv2.resize(frame, dsize=(MODEL_DIM, MODEL_DIM))
            model_start = time.time()
            points = self.model.predict(np.expand_dims(img, axis=0))[0]
            model_end = time.time()

            # Map the points to the original size
            w_ratio = w / MODEL_DIM
            h_ratio = h / MODEL_DIM
            resized_points = []
            for x, y in zip(points[::2], points[1::2]):
                resized_points.append(int(x*w_ratio))
                resized_points.append(int(y*h_ratio))
            resized_points = list(map(lambda el: int(el), resized_points))


            # Place the filters over the frame
            img = frame.copy()
            for i, filter in enumerate(self.filters):
                filter_start = time.time()
                img = filter.place_filter(img, resized_points)
                filter_end = time.time()
                filter_times[i] += filter_end - filter_start

            total_end = time.time()

            # If there are no filters we just draw model predictions
            if self.filters == []:
                for x, y in zip(resized_points[::2], resized_points[1::2]):
                    img = cv2.circle(img, center=(x, y), radius=5, color=(0, 0, 255), thickness=-1)

            # Show the filtered frame
            self.show_frame(img)

            model_time += model_end - model_start
            total_time += total_end - total_start

        video.release()

        # Report the speed of the model and the filters:
        print(f"Average speed of the model predictions: {n_frames / model_time} FPS")
        for filter_name, filter_time in zip(self.filter_names, filter_times):
            print(f"Average speed of the {filter_name} filter placement: {n_frames / filter_time} FPS")
        print(f"Average speed of the whole filtering process: {n_frames / total_time} FPS")

    def initialize_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show_frame(self, img, delay=10):
        cv2.imshow(self.window_name, img)
        cv2.waitKey(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video filtering runner script')
    parser.add_argument("--video", help="Path to the video to place filters over", required=True, action='store')
    parser.add_argument("--glasses", action='store_true', help="If specified, glasses filter is used")
    parser.add_argument("--beard", action='store_true', help="If specified, beard filter is used")
    parser.add_argument("--hat", action='store_true', help="If specified, hat filter is used")

    args = parser.parse_args()

    vf = VideoFilter(glasses=args.glasses, beard=args.beard, hat=args.hat)
    vf.filter_video(args.video)
