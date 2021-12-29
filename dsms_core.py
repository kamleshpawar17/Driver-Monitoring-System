import subprocess
import time

import cv2
import dlib
import numpy as np
from imutils import face_utils
from collections import deque

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = 0.5 * np.float32([[20.0, 20.0, 20.0],
                                 [10.0, 10.0, -10.0],
                                 [10.0, -10.0, -10.0],
                                 [20.0, -20.0, 20.0],
                                 [-20.0, 20.0, 20.0],
                                 [-10.0, 10.0, -10.0],
                                 [-10.0, -10.0, -10.0],
                                 [-20.0, -20.0, 20.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def createQueueWithFill(maxlen, val):
    d = deque(maxlen=maxlen)
    for k in range(maxlen):
        d.append(val)
    return d


class drowsinessAlertBkp():

    def __init__(self, imgsz_fd=96, imgsz_fdlm=300, imgsz_disp=480, CONF_THRSHLD=0.8, frac_above_eye=0.35,
                 scale_bb_width=1.75, aspect_ratio_bb=1.4, fd_temporal_win=2, pose_temporal_win=3, ROBUST_ROT=True,
                 ADJUST_RECT=True, SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True,
                 SHOW_FACE_LANDMARK=True, SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True,
                 SHOW_HIST_EQ=False):
        self.time_at_alarm = time.time()
        self.time_at_alarm_warning = time.time()
        self.ROBUST_ROT = ROBUST_ROT
        self.ADJUST_RECT = ADJUST_RECT
        self.SHOW_ROT_BOX = SHOW_ROT_BOX
        self.BLINK_DETC = BLINK_DETC
        self.PRINT_POSE = PRINT_POSE
        self.SHOW_FACE_LANDMARK = SHOW_FACE_LANDMARK
        self.SHOW_FACE_LANDMARK_5pt = SHOW_FACE_LANDMARK_5pt
        self.SHOW_POSEBOX = SHOW_POSEBOX
        self.HIST_EQ = HIST_EQ
        self.SHOW_HIST_EQ = SHOW_HIST_EQ
        self.CONF_THRSHLD = CONF_THRSHLD
        self.frac_above_eye = frac_above_eye
        self.scale_bb_width = scale_bb_width
        self.aspect_ratio_bb = aspect_ratio_bb
        self.predictor_5lm = dlib.shape_predictor('./lib/models/shape_predictor_5_face_landmarks.dat')
        self.predictor_68lm = dlib.shape_predictor('./lib/models/shape_predictor_68_face_landmarks.dat')
        self.net = cv2.dnn.readNetFromCaffe(
            './lib/models/DSMS_Deploy_FD_MutliScale.prototxt',
            './lib/models/DSMS_MutliScale_kerasTrained_128p_271.caffemodel')
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.imgsz_fd = imgsz_fd
        self.imgsz_fdlm = imgsz_fdlm
        self.imgsz_disp = imgsz_disp
        self.scale_fact = float(self.imgsz_disp) / float(self.imgsz_fdlm)
        self.scale_fact = float(self.imgsz_disp) / float(self.imgsz_fdlm)
        self.facebox = np.zeros((4, 1)).astype("int")
        self.SHOW_FACE_BOX = SHOW_FACE_BOX
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.fd_temporal_win = fd_temporal_win
        self.fd_counter = 0.0
        self.fd_cordinates = np.zeros((4, self.fd_temporal_win))
        self.flag_new_fd = True
        self.pose_counter = 0.0
        self.pose_temporal_win = pose_temporal_win
        self.pose_cordinates = np.zeros((28, self.pose_temporal_win))

    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        image_pts = np.array(image_pts).reshape(1, 28)
        if self.pose_temporal_win == 1:
            image_pts = image_pts
        else:
            image_pts = self.get_smooth_pose(image_pts)
        image_pts = np.float32(tuple(image_pts.reshape(14, 2)))
        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                            dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, euler_angle

    def eye_aspect_ratio(self, eye):
        A = self.euclidean(eye[1], eye[5])
        B = self.euclidean(eye[2], eye[4])
        C = self.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def euclidean(self, x, y):
        return np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)

    def get_smooth_fd_coordinates(self, detections):
        confidence = detections[0][0]
        if confidence > self.CONF_THRSHLD:
            self.fd_cordinates[:, 0:-1] = self.fd_cordinates[:, 1:]
            self.fd_cordinates[:, -1] = np.array(detections[1][0])
            self.fd_counter += 1.0
            self.fd_counter = np.minimum(self.fd_counter, self.fd_temporal_win)
            cenX, cenY, W, H = np.sum(self.fd_cordinates, axis=-1) / self.fd_counter
        else:
            (cenX, cenY, W, H) = detections[1][0]
            self.fd_counter = 0
            self.fd_cordinates = np.zeros((4, self.fd_temporal_win))
        return cenX, cenY, W, H

    def get_smooth_pose(self, lm_points):
        if not self.flag_new_fd:
            self.pose_cordinates[:, 0:-1] = self.pose_cordinates[:, 1:]
            self.pose_cordinates[:, -1] = lm_points
            self.pose_counter += 1.0
            self.pose_counter = np.minimum(self.pose_counter, self.pose_temporal_win)
            lm_points = np.sum(self.pose_cordinates, axis=-1) / self.pose_counter
        else:
            self.pose_counter = 0
            self.pose_cordinates = np.zeros((28, self.pose_temporal_win))
        return lm_points

    def detect_face(self, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (self.imgsz_fd, self.imgsz_fd), interpolation=cv2.INTER_AREA),
                                     1 / 255., swapRB=True)
        self.net.setInput(blob)
        detections = self.net.forward(['loss_ce', 'ip4_eucl'])
        # print(detections)
        confidence = detections[0][0]
        if self.fd_temporal_win == 1:
            cenX, cenY, W, H = detections[1][0]
        else:
            cenX, cenY, W, H = self.get_smooth_fd_coordinates(detections)
        W, H = 1.1 * W, 1.1 * H
        cenX = cenX - 0.1 * W
        startX = int((cenX - W / 2.0) * self.imgsz_fdlm + 0.5)
        startY = int((cenY - H / 2.0) * self.imgsz_fdlm + 0.5)
        endX = int((cenX + W / 2.0) * self.imgsz_fdlm + 0.5)
        endY = int((cenY + H / 2.0) * self.imgsz_fdlm + 0.5)
        face_rects = dlib.rectangle(left=startX, top=startY, right=endX,
                                    bottom=endY)

        startX_1 = int((cenX - W / 2.0) * self.imgsz_disp + 0.5)
        startY_1 = int((cenY - H / 2.0) * self.imgsz_disp + 0.5)
        endX_1 = int((cenX + W / 2.0) * self.imgsz_disp + 0.5)
        endY_1 = int((cenY + H / 2.0) * self.imgsz_disp + 0.5)
        box1 = np.array([startX_1, startY_1, endX_1, endY_1])
        self.facebox = box1.astype("int")
        return face_rects, confidence

    def histeq(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame

    def __call__(self, frame, *args, **kwargs):
        ear = 1.0
        euler_angle = np.zeros((3, 1))
        frame_disp = frame.copy()
        if self.HIST_EQ:
            frame = self.histeq(frame)
        if self.SHOW_HIST_EQ:
            frame_disp = frame

        face_rects, confidence = self.detect_face(frame)

        # ----- set flag to identify if the current fd is newly detected fd or not --- #
        if self.flag_new_fd and (confidence > self.CONF_THRSHLD):
            self.flag_new_fd = False
        if confidence < self.CONF_THRSHLD:
            self.flag_new_fd = True

        if confidence > self.CONF_THRSHLD:
            frame_fdlm = cv2.resize(frame, (self.imgsz_fdlm, self.imgsz_fdlm))

            if self.ROBUST_ROT:
                # ----- Compute 5 landmarks ----- #
                shape_5lm = self.predictor_5lm(frame_fdlm, face_rects)
                shape_5lm = face_utils.shape_to_np(shape_5lm)

                # ----- Compute rotation angle theta ----- #
                indx = [0, 2]
                fit_param = np.polyfit(shape_5lm[indx, 0], shape_5lm[indx, 1], 1)
                theta = np.arctan(fit_param[0]) * 180. / np.pi

                # ----- Rotate original image by theta around centre of face box -----#
                rot_centre_X = (shape_5lm[0, 0] + shape_5lm[2, 0]) / 2
                rot_centre_Y = (shape_5lm[0, 1] + shape_5lm[2, 1]) / 2
                M = cv2.getRotationMatrix2D((rot_centre_X, rot_centre_Y), theta, 1)
                frame_rot = cv2.warpAffine(frame_fdlm, M, (self.imgsz_fdlm, self.imgsz_fdlm))

                # ----- Compute modified rect ----- #
                if self.ADJUST_RECT:
                    W_eye = self.euclidean(shape_5lm[0], shape_5lm[2])
                    W_bb = self.scale_bb_width * W_eye
                    H_bb = self.aspect_ratio_bb * W_bb
                    startX_1 = np.int(rot_centre_X - 0.5 * W_bb + 0.5)
                    endX_1 = np.int(rot_centre_X + 0.5 * W_bb + 0.5)
                    startY_1 = np.int(rot_centre_Y - self.frac_above_eye * H_bb + 0.5)
                    endY_1 = np.int(rot_centre_Y + (1 - self.frac_above_eye) * H_bb + 0.5)
                    face_rects = dlib.rectangle(left=startX_1, top=startY_1, right=endX_1,
                                                bottom=endY_1)
                    if self.SHOW_ROT_BOX:
                        box_pts = []
                        box_pts.append((startX_1, startY_1))
                        box_pts.append((startX_1, endY_1))
                        box_pts.append((endX_1, endY_1))
                        box_pts.append((endX_1, startY_1))
                        box_pts = np.array(box_pts).astype(np.int16)

                # ----- Compute 68 landmarks on rotated image-----#
                shape_68lm = self.predictor_68lm(frame_rot, face_rects)
                shape_68lm = face_utils.shape_to_np(shape_68lm)

                # ----- Rotate 68 landmarks by -theta -----#
                Mt = cv2.getRotationMatrix2D((rot_centre_X, rot_centre_Y), -theta, 1)
                shape = np.zeros((68, 2)).astype(np.int16)
                k = 0
                for (x, y) in shape_68lm:
                    new_pts = self.scale_fact * Mt.dot(np.array([x, y, 1]))
                    shape[k, 0] = np.int(new_pts[0] + 0.5)
                    shape[k, 1] = np.int(new_pts[1] + 0.5)
                    k += 1
                if self.ADJUST_RECT and self.SHOW_ROT_BOX:
                    k = 0
                    for (x, y) in box_pts:
                        new_pts = self.scale_fact * Mt.dot(np.array([x, y, 1]))
                        box_pts[k, 0] = np.int(new_pts[0] + 0.5)
                        box_pts[k, 1] = np.int(new_pts[1] + 0.5)
                        k += 1
            else:
                # ----- Compute 68 landmarks on non rotated image-----#
                shape = self.predictor_68lm(frame_fdlm, face_rects)
                shape = (self.scale_fact * face_utils.shape_to_np(shape) + 0.5).astype(np.int16)

            reprojectdst, euler_angle = self.get_head_pose(shape)

            # ----- Insert 68 face landmark key point ---- #
            if self.SHOW_FACE_LANDMARK:
                for (x, y) in shape:
                    cv2.circle(frame_disp, (x, y), 1, (0, 0, 255), -1)

            # ----- Insert 5 face landmark key point ---- #
            if self.SHOW_FACE_LANDMARK_5pt:
                for (x, y) in shape_5lm:
                    cv2.circle(frame_disp, (x, y), 2, (0, 0, 255), -1)

            # ----- Insert face bounding box ---- #
            if self.ADJUST_RECT and self.SHOW_ROT_BOX:
                box_pts = tuple(map(tuple, box_pts))
                cv2.line(frame_disp, box_pts[0], box_pts[1], (255, 0, 0))
                cv2.line(frame_disp, box_pts[1], box_pts[2], (255, 0, 0))
                cv2.line(frame_disp, box_pts[2], box_pts[3], (255, 0, 0))
                cv2.line(frame_disp, box_pts[3], box_pts[0], (255, 0, 0))

            if self.SHOW_FACE_BOX:
                cv2.rectangle(frame_disp, (self.facebox[0], self.facebox[1]), (self.facebox[2], self.facebox[3]),
                              (0, 0, 255), 2)

            # ----- Insert 3D head pose box ---- #
            if self.SHOW_POSEBOX:
                for start, end in line_pairs:
                    try:
                        cv2.line(frame_disp, reprojectdst[start], reprojectdst[end], (0, 255, 0))
                    except:
                        pass

            if self.PRINT_POSE:
                cv2.putText(frame_disp, "X: " + "{:2.0f}".format(euler_angle[0, 0]), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 0), thickness=2)
                cv2.putText(frame_disp, "Y: " + "{:2.0f}".format(euler_angle[1, 0]), (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 0), thickness=2)
                cv2.putText(frame_disp, "Z: " + "{:2.0f}".format(euler_angle[2, 0]), (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 0), thickness=2)

            # ---- Blink Detection ----- #
            if self.BLINK_DETC:
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

        return frame_disp, ear, euler_angle


class drowsinessAlert():

    def __init__(self, imgsz_fd=96, imgsz_fdlm=300, imgsz_disp=480, CONF_THRSHLD=0.8, frac_above_eye=0.35,
                 scale_bb_width=1.75, aspect_ratio_bb=1.4, fd_temporal_win=2, pose_temporal_win=3, ROBUST_ROT=True,
                 ADJUST_RECT=True, SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True,
                 SHOW_FACE_LANDMARK=True, SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True,
                 SHOW_HIST_EQ=False, HIST_EQ_FDLM=False, isAdjustFdFov=True, showCropBox=True):
        self.time_at_alarm = time.time()
        self.time_at_alarm_warning = time.time()
        self.ROBUST_ROT = ROBUST_ROT
        self.ADJUST_RECT = ADJUST_RECT
        self.SHOW_ROT_BOX = SHOW_ROT_BOX
        self.BLINK_DETC = BLINK_DETC
        self.PRINT_POSE = PRINT_POSE
        self.SHOW_FACE_LANDMARK = SHOW_FACE_LANDMARK
        self.SHOW_FACE_LANDMARK_5pt = SHOW_FACE_LANDMARK_5pt
        self.SHOW_POSEBOX = SHOW_POSEBOX
        self.HIST_EQ = HIST_EQ
        self.HIST_EQ_FDLM = HIST_EQ_FDLM
        self.SHOW_HIST_EQ = SHOW_HIST_EQ
        self.CONF_THRSHLD = CONF_THRSHLD
        self.frac_above_eye = frac_above_eye
        self.scale_bb_width = scale_bb_width
        self.aspect_ratio_bb = aspect_ratio_bb
        self.predictor_5lm = dlib.shape_predictor('./lib/models/shape_predictor_5_face_landmarks.dat')
        self.predictor_68lm = dlib.shape_predictor('./lib/models/shape_predictor_68_face_landmarks.dat')
        self.net = cv2.dnn.readNetFromCaffe(
            './lib/models/DSMS_Deploy_FD_MutliScale.prototxt',
            './lib/models/DSMS_MutliScale_kerasTrained_128p_271.caffemodel')
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.imgsz_fd = imgsz_fd
        self.imgsz_fdlm = imgsz_fdlm
        self.imgsz_disp = imgsz_disp
        self.scale_fact = float(self.imgsz_disp) / float(self.imgsz_fdlm)
        self.facebox = np.zeros((4, 1)).astype("int")
        self.SHOW_FACE_BOX = SHOW_FACE_BOX
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.fd_temporal_win = fd_temporal_win
        self.fd_counter = 0.0
        self.fd_cordinates = np.zeros((4, self.fd_temporal_win))
        self.flag_new_fd = True
        self.pose_counter = 0.0
        self.pose_temporal_win = pose_temporal_win
        self.pose_cordinates = np.zeros((28, self.pose_temporal_win))
        self.isAdjustFdFov = isAdjustFdFov
        if self.isAdjustFdFov:
            self.fd_box = np.load('./lib/fdFovSize.npy')
            self.imgsz_crop = self.fd_box[2] - self.fd_box[0]
            self.imgsz_crop_offsetX = self.fd_box[0]
            self.imgsz_crop_offsetY = self.fd_box[1]
            self.scale_fact_crop = float(self.imgsz_crop) / float(self.imgsz_fdlm)
        else:
            self.imgsz_crop_offsetX = 0
            self.imgsz_crop_offsetY = 0
        self.showCropBox = showCropBox

    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        image_pts = np.array(image_pts).reshape(1, 28)
        if self.pose_temporal_win == 1:
            image_pts = image_pts
        else:
            image_pts = self.get_smooth_pose(image_pts)
        image_pts = np.float32(tuple(image_pts.reshape(14, 2)))
        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                            dist_coeffs)
        reprojectdst = reprojectdst.reshape(8, 2)
        reprojectdst = tuple(map(tuple, reprojectdst))
        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, euler_angle

    def eye_aspect_ratio(self, eye):
        A = self.euclidean(eye[1], eye[5])
        B = self.euclidean(eye[2], eye[4])
        C = self.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def euclidean(self, x, y):
        return np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)

    def get_smooth_fd_coordinates(self, detections):
        confidence = detections[0][0]
        if confidence > self.CONF_THRSHLD:
            self.fd_cordinates[:, 0:-1] = self.fd_cordinates[:, 1:]
            self.fd_cordinates[:, -1] = np.array(detections[1][0])
            self.fd_counter += 1.0
            self.fd_counter = np.minimum(self.fd_counter, self.fd_temporal_win)
            cenX, cenY, W, H = np.sum(self.fd_cordinates, axis=-1) / self.fd_counter
        else:
            (cenX, cenY, W, H) = detections[1][0]
            self.fd_counter = 0
            self.fd_cordinates = np.zeros((4, self.fd_temporal_win))
        return cenX, cenY, W, H

    def get_smooth_pose(self, lm_points):
        if not self.flag_new_fd:
            self.pose_cordinates[:, 0:-1] = self.pose_cordinates[:, 1:]
            self.pose_cordinates[:, -1] = lm_points
            self.pose_counter += 1.0
            self.pose_counter = np.minimum(self.pose_counter, self.pose_temporal_win)
            lm_points = np.sum(self.pose_cordinates, axis=-1) / self.pose_counter
        else:
            self.pose_counter = 0
            self.pose_cordinates = np.zeros((28, self.pose_temporal_win))
        return lm_points

    def detect_face(self, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (self.imgsz_fd, self.imgsz_fd), interpolation=cv2.INTER_AREA),
                                     1 / 255., swapRB=True)
        self.net.setInput(blob)
        detections = self.net.forward(['loss_ce', 'ip4_eucl'])
        # print(detections)
        confidence = detections[0][0]
        if self.fd_temporal_win == 1:
            cenX, cenY, W, H = detections[1][0]
        else:
            cenX, cenY, W, H = self.get_smooth_fd_coordinates(detections)
        # W, H = 1.1 * W, 1.1 * H
        cenX = cenX - 0.1 * W
        startX = int((cenX - W / 2.0) * self.imgsz_fdlm + 0.5)
        startY = int((cenY - H / 2.0) * self.imgsz_fdlm + 0.5)
        endX = int((cenX + W / 2.0) * self.imgsz_fdlm + 0.5)
        endY = int((cenY + H / 2.0) * self.imgsz_fdlm + 0.5)
        face_rects = dlib.rectangle(left=startX, top=startY, right=endX,
                                    bottom=endY)
        if self.isAdjustFdFov:
            startX_1 = int((cenX - W / 2.0) * self.imgsz_crop + 0.5 + self.imgsz_crop_offsetX)
            startY_1 = int((cenY - H / 2.0) * self.imgsz_crop + 0.5 + self.imgsz_crop_offsetX)
            endX_1 = int((cenX + W / 2.0) * self.imgsz_crop + 0.5 + self.imgsz_crop_offsetY)
            endY_1 = int((cenY + H / 2.0) * self.imgsz_crop + 0.5 + self.imgsz_crop_offsetY)
        else:
            startX_1 = int((cenX - W / 2.0) * self.imgsz_disp + 0.5)
            startY_1 = int((cenY - H / 2.0) * self.imgsz_disp + 0.5)
            endX_1 = int((cenX + W / 2.0) * self.imgsz_disp + 0.5)
            endY_1 = int((cenY + H / 2.0) * self.imgsz_disp + 0.5)
        box1 = np.array([startX_1, startY_1, endX_1, endY_1])
        self.facebox = box1.astype("int")
        return face_rects, confidence

    def histeq(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame

    def __call__(self, frame, *args, **kwargs):
        ear = 1.0
        euler_angle = np.zeros((3, 1))
        frame_disp = frame.copy()
        if self.HIST_EQ:
            frame = self.histeq(frame)
        if self.SHOW_HIST_EQ:
            frame_disp = frame
        if self.isAdjustFdFov:
            frame = frame[self.fd_box[1]:self.fd_box[3], self.fd_box[0]:self.fd_box[2], :]
        face_rects, confidence = self.detect_face(frame)

        # ----- set flag to identify if the current fd is newly detected fd or not --- #
        if self.flag_new_fd and (confidence > self.CONF_THRSHLD):
            self.flag_new_fd = False
        if confidence < self.CONF_THRSHLD:
            self.flag_new_fd = True

        if confidence > self.CONF_THRSHLD:
            frame_fdlm = cv2.resize(frame, (self.imgsz_fdlm, self.imgsz_fdlm))
            if self.HIST_EQ_FDLM:
                frame_fdlm = self.histeq(frame_fdlm)

            if self.ROBUST_ROT:
                # ----- Compute 5 landmarks ----- #
                shape_5lm = self.predictor_5lm(frame_fdlm, face_rects)
                shape_5lm = face_utils.shape_to_np(shape_5lm)

                # ----- Compute rotation angle theta ----- #
                indx = [0, 2]
                fit_param = np.polyfit(shape_5lm[indx, 0], shape_5lm[indx, 1], 1)
                theta = np.arctan(fit_param[0]) * 180. / np.pi

                # ----- Rotate original image by theta -----#
                rot_centre_X = (shape_5lm[0, 0] + shape_5lm[2, 0]) / 2
                rot_centre_Y = (shape_5lm[0, 1] + shape_5lm[2, 1]) / 2
                M = cv2.getRotationMatrix2D((rot_centre_X, rot_centre_Y), theta, 1)
                frame_rot = cv2.warpAffine(frame_fdlm, M, (self.imgsz_fdlm, self.imgsz_fdlm))

                # ----- Compute modified rect ----- #
                if self.ADJUST_RECT:
                    W_eye = self.euclidean(shape_5lm[0], shape_5lm[2])
                    W_bb = self.scale_bb_width * W_eye
                    H_bb = self.aspect_ratio_bb * W_bb
                    startX_1 = np.int(rot_centre_X - 0.5 * W_bb + 0.5)
                    endX_1 = np.int(rot_centre_X + 0.5 * W_bb + 0.5)
                    startY_1 = np.int(rot_centre_Y - self.frac_above_eye * H_bb + 0.5)
                    endY_1 = np.int(rot_centre_Y + (1 - self.frac_above_eye) * H_bb + 0.5)
                    face_rects = dlib.rectangle(left=startX_1, top=startY_1, right=endX_1,
                                                bottom=endY_1)
                    if self.SHOW_ROT_BOX:
                        box_pts = []
                        box_pts.append((startX_1, startY_1))
                        box_pts.append((startX_1, endY_1))
                        box_pts.append((endX_1, endY_1))
                        box_pts.append((endX_1, startY_1))
                        box_pts = np.array(box_pts).astype(np.int16)

                # ----- Compute 68 landmarks on rotated image-----#
                shape_68lm = self.predictor_68lm(frame_rot, face_rects)
                shape_68lm = face_utils.shape_to_np(shape_68lm)

                # ----- Rotate 68 landmarks by -theta -----#
                Mt = cv2.getRotationMatrix2D((rot_centre_X, rot_centre_Y), -theta, 1)
                shape = np.zeros((68, 2)).astype(np.int16)
                for k, (x, y) in enumerate(shape_68lm):
                    if self.isAdjustFdFov:
                        new_pts = self.scale_fact_crop * Mt.dot(np.array([x, y, 1]))
                        shape[k, 0] = np.int(new_pts[0] + 0.5 + self.imgsz_crop_offsetX)
                        shape[k, 1] = np.int(new_pts[1] + 0.5 + self.imgsz_crop_offsetY)
                    else:
                        new_pts = self.scale_fact * Mt.dot(np.array([x, y, 1]))
                        shape[k, 0] = np.int(new_pts[0] + 0.5)
                        shape[k, 1] = np.int(new_pts[1] + 0.5)

                if self.ADJUST_RECT and self.SHOW_ROT_BOX:
                    for k, (x, y) in enumerate(box_pts):
                        if self.isAdjustFdFov:
                            new_pts = self.scale_fact_crop * Mt.dot(np.array([x, y, 1]))
                            box_pts[k, 0] = np.int(new_pts[0] + 0.5 + self.imgsz_crop_offsetX)
                            box_pts[k, 1] = np.int(new_pts[1] + 0.5 + self.imgsz_crop_offsetY)
                        else:
                            new_pts = self.scale_fact * Mt.dot(np.array([x, y, 1]))
                            box_pts[k, 0] = np.int(new_pts[0] + 0.5)
                            box_pts[k, 1] = np.int(new_pts[1] + 0.5)

            else:
                # ----- Compute 68 landmarks on non rotated image-----#
                shape = self.predictor_68lm(frame_fdlm, face_rects)
                if self.isAdjustFdFov:
                    shape = (self.scale_fact_crop * face_utils.shape_to_np(shape) + 0.5).astype(np.int16)
                    shape[:, 0] = shape[:, 0] + self.imgsz_crop_offsetX
                    shape[:, 1] = shape[:, 1] + self.imgsz_crop_offsetY
                else:
                    shape = (self.scale_fact * face_utils.shape_to_np(shape) + 0.5).astype(np.int16)

            reprojectdst, euler_angle = self.get_head_pose(shape)

            # ----- Insert 68 face landmark key point ---- #
            if self.SHOW_FACE_LANDMARK:
                for (x, y) in shape:
                    cv2.circle(frame_disp, (x, y), 1, (0, 0, 255), -1)

            # ----- Insert 5 face landmark key point ---- #
            if self.SHOW_FACE_LANDMARK_5pt:
                for (x, y) in shape_5lm:
                    cv2.circle(frame_disp, (int(self.scale_fact_crop * x + self.imgsz_crop_offsetX),
                                            int(self.scale_fact_crop * y + self.imgsz_crop_offsetY)), 2, (0, 255, 255),
                               -1)

            # ----- Insert face bounding box ---- #
            if self.ADJUST_RECT and self.SHOW_ROT_BOX and self.ROBUST_ROT:
                box_pts = tuple(map(tuple, box_pts))
                cv2.line(frame_disp, box_pts[0], box_pts[1], (255, 0, 0))
                cv2.line(frame_disp, box_pts[1], box_pts[2], (255, 0, 0))
                cv2.line(frame_disp, box_pts[2], box_pts[3], (255, 0, 0))
                cv2.line(frame_disp, box_pts[3], box_pts[0], (255, 0, 0))

            if self.SHOW_FACE_BOX:
                cv2.rectangle(frame_disp, (self.facebox[0], self.facebox[1]), (self.facebox[2], self.facebox[3]),
                              (0, 0, 255), 2)
            if self.showCropBox and self.isAdjustFdFov:
                cv2.rectangle(frame_disp, (self.fd_box[0], self.fd_box[1]), (self.fd_box[2], self.fd_box[3]),
                              (0, 255, 255), 2)
            # ----- Insert 3D head pose box ---- #
            if self.SHOW_POSEBOX:
                for start, end in line_pairs:
                    try:
                        cv2.line(frame_disp, reprojectdst[start], reprojectdst[end], (0, 255, 0))
                    except:
                        pass

            if self.PRINT_POSE:
                cv2.putText(frame_disp, "X: " + "{:2.0f}".format(euler_angle[0, 0]), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 0), thickness=2)
                cv2.putText(frame_disp, "Y: " + "{:2.0f}".format(euler_angle[1, 0]), (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 0), thickness=2)
                cv2.putText(frame_disp, "Z: " + "{:2.0f}".format(euler_angle[2, 0]), (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 0), thickness=2)

            # ---- Blink Detection ----- #
            if self.BLINK_DETC:
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

        return frame_disp, ear, euler_angle


class eye_closure_detection():
    '''
    EYE_AR_THRESH: Increasing this increases the sensitivity of eye closure detection, change this with eye size
    EYE_AR_CONSEC_FRAMES: Decreasing this reduces the eye closure detection sensitivity, change this to with varying FPS
    '''

    def __init__(self, EYE_AR_CONSEC_FRAMES=5, EAR_AVG_WIN=5, WARNING_WIN=10, WARNING_TIME=60,
                 ALERT_DUR=10., WAR_DUR=1.0, USE_CALIB=True, DRIVER_ALERT_ALARM=True, DEBUG=False):
        self.EAR_AVG_WIN = EAR_AVG_WIN  # window size for EAR averaging
        self.USE_CALIB = USE_CALIB
        if self.USE_CALIB:
            # Threshold for EAR, EAR<threshold mean eye closure positive
            self.EYE_AR_THRESH = 0.6 * np.load('./lib/ear_thrshld.npy')
        else:
            self.EYE_AR_THRESH = 0.25
        print('EAR Threshold: ', self.EYE_AR_THRESH)
        self.BLINK_COUNT = 0  # Number of positive eye closure
        self.EYE_AR_CONSEC_FRAMES = EYE_AR_CONSEC_FRAMES
        self.WARNING_WIN = WARNING_WIN
        self.WARNING_TIME = WARNING_TIME
        self.ALERT_DUR = ALERT_DUR
        self.WAR_DUR = WAR_DUR
        self.DRIVER_ALERT_ALARM = DRIVER_ALERT_ALARM
        self.DEBUG = DEBUG
        self.ear_queue = createQueueWithFill(maxlen=self.EAR_AVG_WIN, val=1.0)
        self.time_at_alarm = time.time()
        self.time_at_alarm_warning = time.time() - self.ALERT_DUR
        self.warning_array = np.zeros((self.WARNING_WIN))
        self.warning_array.fill(time.time())
        self.warning_array = self.warning_array - (self.WARNING_WIN - np.arange(self.WARNING_WIN)) * self.WARNING_TIME
        self.dispMsgEyeClose = False
        self.dispMsgTIme = time.time()

    def __call__(self, frame, ear, *args, **kwargs):
        self.ear_queue.append(ear)
        ear_arr = np.array(self.ear_queue)
        # If mean of eye AR is < threshold then eye closure is positive
        if np.mean(ear_arr) < self.EYE_AR_THRESH:
            self.BLINK_COUNT += 1
        if self.DEBUG:
            print(self.ear_queue[-1], np.mean(ear_arr), self.BLINK_COUNT, time.time() - self.time_at_alarm,
                  time.time() - self.time_at_alarm_warning)
        # If eye closure is positive for predefined number of frames initiate warning
        if (self.BLINK_COUNT >= self.EYE_AR_CONSEC_FRAMES) and ((time.time() - self.time_at_alarm) > self.WAR_DUR) and (
                (time.time() - self.time_at_alarm_warning) > self.ALERT_DUR):
            subprocess.Popen(['python', './lib/audio/mysound.py'])
            self.dispMsgEyeClose, self.dispMsgTIme = True, time.time()
            self.time_at_alarm = time.time()
            flagEyeClose = True
            if self.DRIVER_ALERT_ALARM:
                self.warning_array = np.roll(self.warning_array, -1)
                self.warning_array[self.WARNING_WIN - 1] = time.time()
                diff = self.warning_array[self.WARNING_WIN - 1] - self.warning_array[0]
                if (diff < self.WARNING_TIME) and (diff > 0):
                    subprocess.Popen(['python', './lib/audio/warning.py'])
                    cv2.putText(frame, "Eye Close Danger", (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 0, 0), thickness=2)
                    self.time_at_alarm_warning = time.time()
                    self.warning_array = np.zeros((self.WARNING_WIN))
                    self.warning_array.fill(time.time())
                    self.warning_array = self.warning_array - (
                            self.WARNING_WIN - np.arange(self.WARNING_WIN)) * self.WARNING_TIME
        else:
            flagEyeClose = False
        if self.dispMsgEyeClose and (time.time() - self.dispMsgTIme) < 2.0:
            cv2.putText(frame, "Eye Close", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
        else:
            self.dispMsgEyeClose = False
        # reset the eye frame counter
        if self.ear_queue[-1] > self.EYE_AR_THRESH:
            self.BLINK_COUNT = 0
        return frame, flagEyeClose


class distraction_detection():
    def __init__(self, buff_size, angle_thrshld_max, angle_thrshld_min, DEBUG=False):
        self.angle_offset = np.load('./lib/angle_offset.npy')
        print('angle offset: ', self.angle_offset)
        self.buff_size = buff_size
        self.angle_thrshld_max = angle_thrshld_max
        self.angle_thrshld_min = angle_thrshld_min
        self.DEBUG = DEBUG
        self.euler_angle_arr = np.zeros((self.buff_size, 1))
        self.count = 0
        self.prev_time = time.time()

    def __call__(self, euler_angle, *args, **kwargs):
        euler_angle = euler_angle - self.angle_offset
        if euler_angle < self.angle_thrshld_min:
            self.euler_angle_arr[self.count] = euler_angle
            self.count += 1
            if self.count == self.buff_size:
                self.count = 0
        else:
            self.euler_angle_arr = np.zeros((self.buff_size, 1))
        diff = time.time() - self.prev_time
        if self.DEBUG:
            print('angle: ', np.mean(self.euler_angle_arr), self.angle_thrshld_max, self.angle_thrshld_min,
                  self.angle_offset)
        if (np.mean(self.euler_angle_arr) < self.angle_thrshld_max) and (diff > 4.0):
            subprocess.Popen(['python', './lib/audio/distraction_sound.py'])
            self.prev_time = time.time()
