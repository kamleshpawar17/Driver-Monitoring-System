import cv2
import time
import numpy as np
from threading import Thread


class USBCamVideoStream:
    def __init__(self, resolution=(320, 240), port=0):
        # initialize the camera and stream
        self.cap = cv2.VideoCapture(port)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.ret = 0

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.cap.isOpened():
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.ret, self.frame = self.cap.read()

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.cap.release()
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class SaveVideoStream:
    def __init__(self, fname, resolution=(640, 480), fps=30.0, warmup=5.0):
        # initialize the video writer
        self.fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.writer = cv2.VideoWriter(fname, self.fourcc, fps, resolution)
        # flag for data flow contraol
        self.frame = np.zeros(resolution, dtype='uint8')
        self.writeFlag = True
        self.fps = fps
        self.warmup = warmup

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        time.sleep(self.warmup)
        # keep looping infinitely until the thread is stopped
        while self.writeFlag:
            time.sleep(1.0 / self.fps)
            self.writer.write(self.frame)
        else:
            self.writer.release()

    def write(self, frame):
        self.frame = frame.copy()

    def stop(self):
        # indicate that the thread should be stopped
        self.writeFlag = False


def SaveVideoBuffer(fname, frames, resolution=(640, 480), fps=30.0):
    # --- Supported combination on mac (MJPG, .avi), (avc1, .mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(fname, fourcc, fps, resolution)
    for frame in frames:
        writer.write(frame)
    writer.release()

