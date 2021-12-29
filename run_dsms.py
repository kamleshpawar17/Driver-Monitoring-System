import argparse
import time
from collections import deque
from datetime import datetime
from multiprocessing.pool import ThreadPool
from threading import Thread

import cv2
import numpy as np

from dsms_core import eye_closure_detection, drowsinessAlert, distraction_detection
from videoStreamUSBCam import USBCamVideoStream, SaveVideoBuffer


def get_cropped_frame(frame, imgsz_disp):
    (h, w) = frame.shape[:2]
    y0 = np.int16(h / 2 - imgsz_disp / 2)
    y1 = np.int16(h / 2 + imgsz_disp / 2)
    x0 = np.int16(w / 2 - imgsz_disp / 2)
    x1 = np.int16(w / 2 + imgsz_disp / 2)
    frame = frame[y0:y1, x0:x1, :]
    return frame


def getFdFovCropScale(frame, fd_box, imgsz_disp):
    frame = frame[fd_box[0]:fd_box[1], fd_box[2]:fd_box[3], :]
    frame = cv2.resize(frame, (imgsz_disp, imgsz_disp), interpolation=cv2.INTER_AREA)
    return frame


def calibrationBkp(imgsz_disp, imgsz_fd, imgsz_fdlm, NUM=50, AVG_WIN=256):
    '''
    This function does the calibration scan to find the EAR and viewing angle offset
    :param imgsz_disp: size of the image to display
    :param imgsz_fd: size of the image for face detection
    :param imgsz_fdlm: size of the image for landmark detection
    :param NUM: Number of frames to calculate frame rate
    :return: None
    '''
    # ----- USB cam setup ---- #
    vs = USBCamVideoStream(resolution=(imgsz_disp, imgsz_disp), port=0).start()
    time.sleep(2.0)

    drowsinessAlert_obj0 = drowsinessAlert(ADJUST_RECT=True, PRINT_POSE=True, BLINK_DETC=True,
                                           SHOW_FACE_BOX=False, SHOW_POSEBOX=True, SHOW_ROT_BOX=False,
                                           SHOW_FACE_LANDMARK=False, HIST_EQ=True, ROBUST_ROT=True,
                                           imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                           isAdjustFdFov=False)

    ear_calib_arr = np.zeros((AVG_WIN, 1))
    angle_offset_arr = np.zeros((AVG_WIN, 1))
    count = 0
    count_fps = 0
    fps = 0
    start_flag = False
    while True:
        if count_fps == 0:
            start_fps = time.time()
        frame = vs.read()
        frame = get_cropped_frame(frame, imgsz_disp)
        frame_out, ear, euler_angle = drowsinessAlert_obj0(frame)
        count_fps += 1
        if count_fps == NUM:
            count_fps = 0
            end_fps = time.time()
            elapsed_time = end_fps - start_fps
            fps = np.ceil(NUM / elapsed_time)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            start_flag = True
            count = 0

        if start_flag:
            cv2.putText(frame_out, 'Calibration in progress ... ', (30, 40), cv2.FONT_ITALIC, 1.0,
                        (0, 0, 255), 1)
            ear_calib_arr[count] = ear
            angle_offset_arr[count] = euler_angle[1, 0]
            count += 1
            if count == AVG_WIN:
                ear_mean = np.mean(ear_calib_arr)
                np.save('./lib/ear_thrshld', ear_mean)
                angle_offset_mean = np.mean(angle_offset_arr)
                np.save('./lib/angle_offset', angle_offset_mean)
                start_flag = False
        else:
            cv2.putText(frame_out, 'Press s to start calibration', (30, 40), cv2.FONT_ITALIC, 1.0,
                        (255, 0, 0), 1)
            cv2.putText(frame_out, 'Press q to quit', (30, 100), cv2.FONT_ITALIC, 1.0, (255, 0, 0), 1)

        cv2.putText(frame_out, 'FPS: ' + str(fps), (300, 40), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
        cv2.imshow("calibration", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.stop()
    cv2.destroyAllWindows()


def calibrationFdFov(vs, AVG_WIN_FDFOV=64, FovFdRatio=1.5, imgsz_fd=96, imgsz_fdlm=300, imgsz_disp=480,
                     CONF_THRSHLD=0.8, frac_above_eye=0.35,
                     scale_bb_width=1.75, aspect_ratio_bb=1.4, fd_temporal_win=2, pose_temporal_win=3, ROBUST_ROT=True,
                     ADJUST_RECT=True, SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True,
                     SHOW_FACE_LANDMARK=True, SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True,
                     SHOW_HIST_EQ=False, HIST_EQ_FDLM=False):
    '''
    This function does the calibration scan to find the EAR and viewing angle offset
    :param imgsz_disp: size of the image to display
    :param imgsz_fd: size of the image for face detection
    :param imgsz_fdlm: size of the image for landmark detection
    :param NUM: Number of frames to calculate frame rate
    :return: None
    '''
    drowsinessAlert_obj0 = drowsinessAlert(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                           CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                           scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                           fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                           ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX,
                                           SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                           SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                           SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt, SHOW_POSEBOX=SHOW_POSEBOX,
                                           HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM, SHOW_HIST_EQ=SHOW_HIST_EQ,
                                           isAdjustFdFov=False)

    fd_box_arr = np.zeros((AVG_WIN_FDFOV, 4))  # (startY, endY, startX, endX)
    for count in range(AVG_WIN_FDFOV):
        frame = vs.read()
        frame = get_cropped_frame(frame, imgsz_disp)
        frame_out, ear, euler_angle = drowsinessAlert_obj0(frame)
        cv2.putText(frame_out, 'Calibration in progress ... ', (30, 40), cv2.FONT_ITALIC, 1.0,
                    (0, 0, 255), 1)
        cv2.imshow("calibration", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fd_box_arr[count, :] = drowsinessAlert_obj0.facebox.copy()
    fd_box_mean = np.mean(fd_box_arr, axis=0)
    fdXCentre = 0.5 * (fd_box_mean[0] + fd_box_mean[2])
    fdYCentre = 0.5 * (fd_box_mean[1] + fd_box_mean[3])
    fdWidth = FovFdRatio * (fd_box_mean[2] - fd_box_mean[0])
    fdHeight = FovFdRatio * (fd_box_mean[3] - fd_box_mean[1])
    fdFovSize = np.maximum(fdWidth, fdHeight)
    startXFdFov = np.maximum(0, fdXCentre - fdFovSize / 2)
    endXFdFov = np.minimum(imgsz_disp, fdXCentre + fdFovSize / 2)
    startYFdFov = np.maximum(0, fdYCentre - fdFovSize / 2)
    endYFdFov = np.minimum(imgsz_disp, fdYCentre + fdFovSize / 2)
    fdFov = np.array([startXFdFov, startYFdFov, endXFdFov, endYFdFov], dtype='int')
    np.save('./lib/fdFovSize', fdFov)
    time.sleep(0.5)


def calibrationEarYawOffset(vs, AVG_WIN=256, imgsz_fd=96, imgsz_fdlm=300, imgsz_disp=480, CONF_THRSHLD=0.8,
                            frac_above_eye=0.35,
                            scale_bb_width=1.75, aspect_ratio_bb=1.4, fd_temporal_win=2, pose_temporal_win=3,
                            ROBUST_ROT=True,
                            ADJUST_RECT=True, SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True,
                            SHOW_FACE_LANDMARK=True, SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True,
                            SHOW_HIST_EQ=False, HIST_EQ_FDLM=False):
    '''
    This function does the calibration scan to find the EAR and viewing angle offset
    :param imgsz_disp: size of the image to display
    :param imgsz_fd: size of the image for face detection
    :param imgsz_fdlm: size of the image for landmark detection
    :param NUM: Number of frames to calculate frame rate
    :return: None
    '''
    drowsinessAlert_obj0 = drowsinessAlert(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                           CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                           scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                           fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                           ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX,
                                           SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                           SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                           SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt, SHOW_POSEBOX=SHOW_POSEBOX,
                                           HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM, SHOW_HIST_EQ=SHOW_HIST_EQ,
                                           isAdjustFdFov=True, showCropBox=True)

    ear_calib_arr = np.zeros((AVG_WIN, 1))
    angle_offset_arr = np.zeros((AVG_WIN, 1))
    for count in range(AVG_WIN):
        frame = vs.read()
        frame = get_cropped_frame(frame, imgsz_disp)
        frame_out, ear, euler_angle = drowsinessAlert_obj0(frame)
        cv2.putText(frame_out, 'Calibration in progress ... ', (30, 40), cv2.FONT_ITALIC, 1.0,
                    (0, 0, 255), 1)
        cv2.imshow("calibration", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ear_calib_arr[count] = ear
        angle_offset_arr[count] = euler_angle[1, 0]
    ear_mean = np.mean(ear_calib_arr)
    np.save('./lib/ear_thrshld', ear_mean)
    angle_offset_mean = np.mean(angle_offset_arr)
    np.save('./lib/angle_offset', angle_offset_mean)


def calibration(AVG_WIN=256, AVG_WIN_FDFOV=64, FovFdRatio=1.5, imgsz_fd=96, imgsz_fdlm=300,
                imgsz_disp=480, CONF_THRSHLD=0.8, frac_above_eye=0.35, scale_bb_width=1.75, aspect_ratio_bb=1.4,
                fd_temporal_win=2, pose_temporal_win=3, ROBUST_ROT=True, ADJUST_RECT=True, SHOW_FACE_BOX=False,
                SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True, SHOW_FACE_LANDMARK=True,
                SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True, SHOW_HIST_EQ=False, HIST_EQ_FDLM=False):
    '''
    This function does the calibration scan to find the EAR and viewing angle offset
    :param imgsz_disp: size of the image to display
    :param imgsz_fd: size of the image for face detection
    :param imgsz_fdlm: size of the image for landmark detection
    :param NUM: Number of frames to calculate frame rate
    :return: None
    '''
    # ----- USB cam setup ---- #
    vs = USBCamVideoStream(resolution=(imgsz_disp, imgsz_disp), port=0).start()
    time.sleep(2.0)

    start_flag = False
    while True:
        frame = vs.read()
        frame = get_cropped_frame(frame, imgsz_disp)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            start_flag = True

        if start_flag:
            calibrationFdFov(vs, AVG_WIN_FDFOV=AVG_WIN_FDFOV, FovFdRatio=FovFdRatio, imgsz_fd=imgsz_fd,
                             imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                             CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                             scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                             fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                             ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX,
                             SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                             SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                             SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt, SHOW_POSEBOX=SHOW_POSEBOX,
                             HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM, SHOW_HIST_EQ=SHOW_HIST_EQ)
            calibrationEarYawOffset(vs, AVG_WIN=AVG_WIN, imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm,
                                    imgsz_disp=imgsz_disp,
                                    CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                    scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                    fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                    ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX,
                                    SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                    SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                    SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt, SHOW_POSEBOX=SHOW_POSEBOX,
                                    HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM, SHOW_HIST_EQ=SHOW_HIST_EQ)
            start_flag = False
        else:
            cv2.putText(frame, 'Press s to start calibration', (30, 40), cv2.FONT_ITALIC, 1.0,
                        (255, 0, 0), 1)
            cv2.putText(frame, 'Press q to quit', (30, 100), cv2.FONT_ITALIC, 1.0, (255, 0, 0), 1)

        cv2.imshow("calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.stop()
    cv2.destroyAllWindows()


def run_single_thread(imgsz_fd=96, imgsz_fdlm=300, imgsz_disp=480, CONF_THRSHLD=0.8, frac_above_eye=0.35,
                      scale_bb_width=1.75, aspect_ratio_bb=1.4, fd_temporal_win=2, pose_temporal_win=3, ROBUST_ROT=True,
                      ADJUST_RECT=True, SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True,
                      SHOW_FACE_LANDMARK=True, SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True,
                      SHOW_HIST_EQ=False, HIST_EQ_FDLM=False, SHOW_IMG=1, NUM=50, EAR_AVG_WIN=5,
                      EYE_AR_CONSEC_FRAMES=2, SAVE_VID=False, DISTR_DET=True, buff_size=12, angle_thrshld_max=-25.0,
                      angle_thrshld_min=-5, isAdjustFdFov=True, showCropBox=True):
    # --- Initialize video writer object ---- #
    framesBuffer = deque(maxlen=150)
    delTime = 2.0
    # ---- Instantiate detection objects and Initialize  threads for the first time ---- #
    drowsinessAlert_obj = drowsinessAlert(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                          CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                          scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                          fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                          ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX,
                                          SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                          SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                          SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt, SHOW_POSEBOX=SHOW_POSEBOX,
                                          HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM, SHOW_HIST_EQ=SHOW_HIST_EQ,
                                          isAdjustFdFov=isAdjustFdFov, showCropBox=showCropBox)

    # ----- Eye closure detection ----- #
    ear_obj = eye_closure_detection(EYE_AR_CONSEC_FRAMES=EYE_AR_CONSEC_FRAMES, DEBUG=False, EAR_AVG_WIN=EAR_AVG_WIN,
                                    DRIVER_ALERT_ALARM=False)
    # ----- Distraction detection ---- #
    if DISTR_DET:
        distraction_obj = distraction_detection(buff_size=buff_size, angle_thrshld_max=angle_thrshld_max,
                                                angle_thrshld_min=angle_thrshld_min, DEBUG=False)
    # ----- Initilize variables ---- #
    start_fps, timeEyeclose = time.time(), time.time() - delTime
    count, fps, flagWrite = 0, 0, False,

    # ----- USB cam setup ---- #
    vs = USBCamVideoStream(resolution=(imgsz_disp, imgsz_disp), port=0).start()
    time.sleep(2.0)
    while True:
        frame_pi = vs.read()
        frame_pi = get_cropped_frame(frame_pi, imgsz_disp)
        # if isAdjustFdFov:
        #     frame_pi = getFdFovCropScale(frame_pi, fd_box, imgsz_disp)
        frame_out, ear, euler_angle = drowsinessAlert_obj(frame_pi)
        # ----- FPS computation ----- #
        count += 1
        if count == NUM:
            end_fps = time.time()
            elapsed_time = end_fps - start_fps
            start_fps = time.time()
            fps = np.ceil(NUM / elapsed_time)
            count = 0

        cv2.putText(frame_out, 'FPS: ' + str(fps), (300, 40), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
        # ---- Drowsiness Detection ---- #
        frame_out, flagEyeClose = ear_obj(frame_out, ear)
        if flagEyeClose and (time.time() - timeEyeclose) > delTime:
            timeEyeclose = time.time()
            flagWrite = True

        # ---- store frames buffer and write to file --- #
        if SAVE_VID:
            framesBuffer.append(frame_out)
            if flagWrite and (time.time() - timeEyeclose) > delTime:
                flagWrite = False
                Thread(target=SaveVideoBuffer, args=(
                    './savedVideo/unsend/' + datetime.now().strftime("Date-%b:%d:%Y_Time-%H.%M.%S") + '.mp4',
                    framesBuffer.copy(), (imgsz_disp, imgsz_disp),
                    25.0)).start()

        # --- Distraction Detection ---- #
        if DISTR_DET:
            distraction_obj(euler_angle[1, 0])
        if SHOW_IMG:
            cv2.imshow("DSMS", frame_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vs.stop()
                break
        else:
            print('FPS:', fps)


def run_multi_thread(imgsz_fd=96, imgsz_fdlm=300, imgsz_disp=480, CONF_THRSHLD=0.8, frac_above_eye=0.35,
                     scale_bb_width=1.75, aspect_ratio_bb=1.4, fd_temporal_win=2, pose_temporal_win=3, ROBUST_ROT=True,
                     ADJUST_RECT=True, SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True,
                     SHOW_FACE_LANDMARK=True, SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True,
                     SHOW_HIST_EQ=False, HIST_EQ_FDLM=False, SHOW_IMG=1, NUM=50, EAR_AVG_WIN=5,
                     EYE_AR_CONSEC_FRAMES=2,  SAVE_VID=False, DISTR_DET=True, buff_size=12, angle_thrshld_max=-25.0,
                     angle_thrshld_min=-5, isAdjustFdFov=True, showCropBox=True):
    # --- Initialize video writer object ---- #
    framesBuffer = deque(maxlen=150)
    delTime = 2.0

    # ----- Drowsiness detection ----- #
    ear_obj = eye_closure_detection(EYE_AR_CONSEC_FRAMES=EYE_AR_CONSEC_FRAMES, DEBUG=False, EAR_AVG_WIN=EAR_AVG_WIN,
                                    DRIVER_ALERT_ALARM=False)
    euler_angle = np.zeros((3, 1))
    # ----- Distraction detection ---- #
    if DISTR_DET:
        distraction_obj = distraction_detection(buff_size=buff_size, angle_thrshld_max=angle_thrshld_max,
                                                angle_thrshld_min=angle_thrshld_min, DEBUG=False)
    # ----- USB cam setup ---- #
    vs = USBCamVideoStream(resolution=(imgsz_disp, imgsz_disp), port=0).start()
    time.sleep(2.0)
    # ---- Instantiate detection objects and Initialize  threads for the first time ---- #
    if NUM_PROC >= 1:
        drowsinessAlert_obj_0 = drowsinessAlert(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                                CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                                scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                                fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                                ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT,
                                                SHOW_FACE_BOX=SHOW_FACE_BOX, SHOW_ROT_BOX=SHOW_ROT_BOX,
                                                BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                                SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                                SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt,
                                                SHOW_POSEBOX=SHOW_POSEBOX, HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM,
                                                SHOW_HIST_EQ=SHOW_HIST_EQ, isAdjustFdFov=isAdjustFdFov,
                                                showCropBox=showCropBox)
        pool_0 = ThreadPool(processes=1)
        frame_0 = vs.read()
        frame_0 = get_cropped_frame(frame_0, imgsz_disp)
        results_0 = pool_0.apply_async(drowsinessAlert_obj_0, args=(frame_0,))
    if NUM_PROC >= 2:
        drowsinessAlert_obj_1 = drowsinessAlert(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                                CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                                scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                                fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                                ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT,
                                                SHOW_FACE_BOX=SHOW_FACE_BOX, SHOW_ROT_BOX=SHOW_ROT_BOX,
                                                BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                                SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                                SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt,
                                                SHOW_POSEBOX=SHOW_POSEBOX, HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM,
                                                SHOW_HIST_EQ=SHOW_HIST_EQ, isAdjustFdFov=isAdjustFdFov,
                                                showCropBox=showCropBox)
        pool_1 = ThreadPool(processes=1)
        frame_1 = vs.read()
        frame_1 = get_cropped_frame(frame_1, imgsz_disp)
        results_1 = pool_1.apply_async(drowsinessAlert_obj_1, args=(frame_1,))
    if NUM_PROC >= 3:
        drowsinessAlert_obj_2 = drowsinessAlert(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                                CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                                scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                                fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                                ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT,
                                                SHOW_FACE_BOX=SHOW_FACE_BOX, SHOW_ROT_BOX=SHOW_ROT_BOX,
                                                BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                                SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                                SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt,
                                                SHOW_POSEBOX=SHOW_POSEBOX, HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM,
                                                SHOW_HIST_EQ=SHOW_HIST_EQ, isAdjustFdFov=isAdjustFdFov,
                                                showCropBox=showCropBox)
        pool_2 = ThreadPool(processes=1)
        frame_2 = vs.read()
        frame_2 = get_cropped_frame(frame_2, imgsz_disp)
        results_2 = pool_2.apply_async(drowsinessAlert_obj_2, args=(frame_2,))
    if NUM_PROC >= 4:
        drowsinessAlert_obj_3 = drowsinessAlert(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp,
                                                CONF_THRSHLD=CONF_THRSHLD, frac_above_eye=frac_above_eye,
                                                scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                                                fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                                                ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT,
                                                SHOW_FACE_BOX=SHOW_FACE_BOX, SHOW_ROT_BOX=SHOW_ROT_BOX,
                                                BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                                                SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK,
                                                SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt,
                                                SHOW_POSEBOX=SHOW_POSEBOX, HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM,
                                                SHOW_HIST_EQ=SHOW_HIST_EQ, isAdjustFdFov=isAdjustFdFov,
                                                showCropBox=showCropBox)
        pool_3 = ThreadPool(processes=1)
        frame_3 = vs.read()
        frame_3 = get_cropped_frame(frame_3, imgsz_disp)
        results_3 = pool_3.apply_async(drowsinessAlert_obj_3, args=(frame_3,))

    # ----- Initilize variables ---- #
    start_fps, timeEyeclose = time.time(), time.time() - delTime
    count, fps, flagWrite, indx, ear = 0, 0, False, 0, 1.0

    # ----- Loop for detection ------ #
    while True:
        if count == 0:
            start_fps = time.time()
        if indx == 0:
            output = results_0.get()
            frame_out = output[0]
            ear = output[1]
            euler_angle = output[2]
            frame_0 = vs.read()
            frame_0 = get_cropped_frame(frame_0, imgsz_disp)
            results_0 = pool_0.apply_async(drowsinessAlert_obj_0, args=(frame_0,))
        if indx == 1:
            output = results_1.get()
            frame_out = output[0]
            ear = output[1]
            euler_angle = output[2]
            frame_1 = vs.read()
            frame_1 = get_cropped_frame(frame_1, imgsz_disp)
            results_1 = pool_1.apply_async(drowsinessAlert_obj_1, args=(frame_1,))
        if indx == 2:
            output = results_2.get()
            frame_out = output[0]
            ear = output[1]
            euler_angle = output[2]
            frame_2 = vs.read()
            frame_2 = get_cropped_frame(frame_2, imgsz_disp)
            results_2 = pool_2.apply_async(drowsinessAlert_obj_2, args=(frame_2,))
        if indx == 3:
            output = results_3.get()
            frame_out = output[0]
            ear = output[1]
            euler_angle = output[2]
            frame_3 = vs.read()
            frame_3 = get_cropped_frame(frame_3, imgsz_disp)
            results_3 = pool_3.apply_async(drowsinessAlert_obj_3, args=(frame_3,))

        count += 1
        indx += 1
        if indx == NUM_PROC:
            indx = 0
        # ----- FPS computation ----- #
        if count == NUM:
            end_fps = time.time()
            elapsed_time = end_fps - start_fps
            fps = np.ceil(NUM / elapsed_time)
            count = 0

        cv2.putText(frame_out, 'FPS: ' + str(fps), (300, 40), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
        # ---- Drowsiness Detection ---- #
        frame_out, flagEyeClose = ear_obj(frame_out, ear)
        if flagEyeClose and (time.time() - timeEyeclose) > delTime:
            timeEyeclose = time.time()
            flagWrite = True

        # ---- store frames buffer and write to file --- #
        if SAVE_VID:
            framesBuffer.append(frame_out)
            if flagWrite and (time.time() - timeEyeclose) > delTime:
                flagWrite = False
                Thread(target=SaveVideoBuffer, args=(
                    './savedVideo/unsend/' + datetime.now().strftime("Date-%b:%d:%Y_Time-%H.%M.%S") + '.mp4',
                    framesBuffer.copy(), (imgsz_disp, imgsz_disp),
                    25.0)).start()

        # --- Distraction Detection ---- #
        if DISTR_DET:
            distraction_obj(euler_angle[1, 0])
        if SHOW_IMG:
            cv2.imshow("DSMS", frame_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vs.stop()
                break
        else:
            print('FPS: ', fps)


def run(imgsz_fd=96, imgsz_fdlm=300, imgsz_disp=480, CONF_THRSHLD=0.8, frac_above_eye=0.35, scale_bb_width=1.75,
        aspect_ratio_bb=1.4, fd_temporal_win=2, pose_temporal_win=3, ROBUST_ROT=True, ADJUST_RECT=True,
        SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True, SHOW_FACE_LANDMARK=True,
        SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True, HIST_EQ_FDLM=False, SHOW_HIST_EQ=False,
        SHOW_IMG=1, NUM=50, EAR_AVG_WIN=5, EYE_AR_CONSEC_FRAMES=2, SAVE_VID=False, DISTR_DET=True, MULTIPROC_FLAG=False,
        buff_size=12, angle_thrshld_max=-25.0, angle_thrshld_min=-5, isAdjustFdFov=True, showCropBox=True):
    '''
    This function runs loop for monitoring driver state
    :param imgsz_disp: size of the image to display
    :param imgsz_fd: size of the image for face detection
    :param imgsz_fdlm: size of the image for landmark detection
    :param MULTIPROC_FLAG: Boolean for using mutiprocessing
    :param NUM_PROC: Number of parallel threads for multiprocesing
    :param NUM: Number of frames to calculate frame rate
    :param EAR_AVG_WIN: window size for eye aspect ratio averaging
    :param EYE_AR_CONSEC_FRAMES: Number of consecutive frames to initiate eye closure alarm
    :return: None
    '''
    if MULTIPROC_FLAG:
        run_multi_thread(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp, CONF_THRSHLD=CONF_THRSHLD,
                         frac_above_eye=frac_above_eye, scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                         fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                         ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX,
                         SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                         SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK, SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt,
                         SHOW_POSEBOX=SHOW_POSEBOX, HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM,
                         SHOW_HIST_EQ=SHOW_HIST_EQ, SHOW_IMG=SHOW_IMG, NUM=NUM, EAR_AVG_WIN=EAR_AVG_WIN,
                         EYE_AR_CONSEC_FRAMES=EYE_AR_CONSEC_FRAMES, SAVE_VID=SAVE_VID,
                         DISTR_DET=DISTR_DET, buff_size=buff_size, angle_thrshld_max=angle_thrshld_max,
                         angle_thrshld_min=angle_thrshld_min, isAdjustFdFov=isAdjustFdFov, showCropBox=showCropBox)
    else:
        run_single_thread(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp, CONF_THRSHLD=CONF_THRSHLD,
                          frac_above_eye=frac_above_eye, scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                          fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win,
                          ROBUST_ROT=ROBUST_ROT, ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX,
                          SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC, PRINT_POSE=PRINT_POSE,
                          SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK, SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt,
                          SHOW_POSEBOX=SHOW_POSEBOX, HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM,
                          SHOW_HIST_EQ=SHOW_HIST_EQ, SHOW_IMG=SHOW_IMG, NUM=NUM, EAR_AVG_WIN=EAR_AVG_WIN,
                          EYE_AR_CONSEC_FRAMES=EYE_AR_CONSEC_FRAMES, SAVE_VID=SAVE_VID,
                          DISTR_DET=DISTR_DET, buff_size=buff_size, angle_thrshld_max=angle_thrshld_max,
                          angle_thrshld_min=angle_thrshld_min, isAdjustFdFov=isAdjustFdFov, showCropBox=showCropBox)


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--showvideo", type=int, default=True)
ap.add_argument("-m", "--multiproc", type=int, default=True)
ap.add_argument("-np", "--numproc", type=int, default=2)
ap.add_argument("-c", "--calib", type=int, default=True)
ap.add_argument("-sav", "--savevideo", type=int, default=False)
args = vars(ap.parse_args())
if __name__ == '__main__':
    # --- Command line arguments --- #
    MULTIPROC_FLAG = args['multiproc']
    NUM_PROC = args['numproc']
    SHOW_IMG = args['showvideo']
    RUN_CALIB = args['calib']
    SAVE_VID = args['savevideo']

    # ----- Params FD ---- #
    imgsz_disp = 480  # Image size to display
    imgsz_fd = 128  # Image size for face detection
    imgsz_fdlm = 128  # Image size for face landmark detection
    CONF_THRSHLD = 0.8  # confidence threshold for face detection
    frac_above_eye = 0.35  # fraction of bb above the eye line
    scale_bb_width = 1.75  # face bb width = eye-width * scale_bb_width
    aspect_ratio_bb = 1.4  # ratio of face bb: height/width
    fd_temporal_win = 3  # averaging window for face bb coordinates
    pose_temporal_win = 3  # averaging window for 14 face landmark points
    NUM = 50  # number of frames to compute the average frame rate
    ROBUST_ROT = True  # if true enable robust rotation of face
    ADJUST_RECT = True  # if true adjust the face box of the rotated face
    SHOW_FACE_BOX = False  # if true, show face box
    SHOW_ROT_BOX = True  # if true, show rotated face box
    BLINK_DETC = True  # if true, enable eye drowsiness detection
    PRINT_POSE = True  # if true, print roll, pitch and yaw of the face pose in image
    SHOW_FACE_LANDMARK = True  # if true, show 68 face landmark points
    SHOW_FACE_LANDMARK_5pt = False  # if true, show 5 face landmark points around eye
    SHOW_POSEBOX = True  # if true, show the face pose box
    HIST_EQ = True  # if true, enable histogram equalization of image
    HIST_EQ_FDLM = not HIST_EQ
    SHOW_HIST_EQ = True  # if true, show histogram equalized image
    isAdjustFdFov = True  # if True, crop the image around face for detection
    showCropBox = True  # if true, show the ffd fov box

    # ----- Params Eye closure detection ---- #
    EYE_AR_CONSEC_FRAMES = 10  # Number of consecutive eye clousre frames after which warning is activated
    EAR_AVG_WIN = 10  # averaging window for eye aspect ratio
    WARNING_WIN = 10  # raise an alarm to driver if WARNING_WIN times eye closure is detected within in WARNING_TIME
    WARNING_TIME = 60
    ALERT_DUR = 10.  # if alert warning initiated, the next warning will not be initiated before ALERT_DUR
    WAR_DUR = 1.0  # if eye closure warning initiated, the next warning will not be initiated before ALERT_DUR
    USE_CALIB = True  # if true, use the stored eye aspect ratio and head pose from the calibration scan
    DRIVER_ALERT_ALARM = False  # if true, enable the driver alert alarm
    DEBUG = False  # if true print the debug information

    # ---- Distraction detection ---- #
    buff_size = 12  # averaging window size for the head pose yaw angle
    angle_thrshld_max = -25.0  # yaw angle threshold to initiate warning (left)
    angle_thrshld_min = -5  # yaw angle threshold to initiate warning (right) - not implemented
    DISTR_DET = True  # if true, enable distraction warning

    # ---- Calibration params ----- #
    AVG_WIN = 256  # number of frame to compute the average eye-aspect ratio and yaw angle offset
    AVG_WIN_FDFOV = 64  # number of frame to compute fd FOV
    FovFdRatio = 2.0  # ratio of fdFov and fd bb

    if RUN_CALIB:
        calibration(AVG_WIN=AVG_WIN, AVG_WIN_FDFOV=AVG_WIN_FDFOV, FovFdRatio=FovFdRatio, imgsz_fd=imgsz_fd,
                    imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp, CONF_THRSHLD=CONF_THRSHLD,
                    frac_above_eye=frac_above_eye, scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
                    fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win, ROBUST_ROT=True,
                    ADJUST_RECT=True, SHOW_FACE_BOX=False, SHOW_ROT_BOX=False, BLINK_DETC=True, PRINT_POSE=True,
                    SHOW_FACE_LANDMARK=True, SHOW_FACE_LANDMARK_5pt=False, SHOW_POSEBOX=True, HIST_EQ=True,
                    SHOW_HIST_EQ=False, HIST_EQ_FDLM=False)

    run(imgsz_fd=imgsz_fd, imgsz_fdlm=imgsz_fdlm, imgsz_disp=imgsz_disp, CONF_THRSHLD=CONF_THRSHLD,
        frac_above_eye=frac_above_eye, scale_bb_width=scale_bb_width, aspect_ratio_bb=aspect_ratio_bb,
        fd_temporal_win=fd_temporal_win, pose_temporal_win=pose_temporal_win, ROBUST_ROT=ROBUST_ROT,
        ADJUST_RECT=ADJUST_RECT, SHOW_FACE_BOX=SHOW_FACE_BOX, SHOW_ROT_BOX=SHOW_ROT_BOX, BLINK_DETC=BLINK_DETC,
        PRINT_POSE=PRINT_POSE, SHOW_FACE_LANDMARK=SHOW_FACE_LANDMARK, SHOW_FACE_LANDMARK_5pt=SHOW_FACE_LANDMARK_5pt,
        SHOW_POSEBOX=SHOW_POSEBOX, HIST_EQ=HIST_EQ, HIST_EQ_FDLM=HIST_EQ_FDLM, SHOW_HIST_EQ=SHOW_HIST_EQ,
        SHOW_IMG=SHOW_IMG, NUM=NUM, EAR_AVG_WIN=EAR_AVG_WIN, EYE_AR_CONSEC_FRAMES=EYE_AR_CONSEC_FRAMES, 
        SAVE_VID=SAVE_VID, DISTR_DET=DISTR_DET, MULTIPROC_FLAG=MULTIPROC_FLAG, buff_size=buff_size, angle_thrshld_max=angle_thrshld_max,
        angle_thrshld_min=angle_thrshld_min, isAdjustFdFov=isAdjustFdFov, showCropBox=showCropBox)
