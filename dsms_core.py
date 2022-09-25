import time
from collections import deque
from datetime import datetime
from multiprocessing.pool import ThreadPool
from threading import Thread

import cv2
import numpy as np

from dsms_analytics import eye_closure_detection, drowsinessAlert, distraction_detection
from video_stream_usb_cam import USBCamVideoStream, SaveVideoBuffer


def get_cropped_frame(frame, imgsz_disp):
    """function to crop the image frame

    Args:
        frame (cv::Mat): image matrix
        imgsz_disp (int): image size

    Returns:
        cv::Mat: image matrix
    """
    (h, w) = frame.shape[:2]
    y0 = np.int16(h / 2 - imgsz_disp / 2)
    y1 = np.int16(h / 2 + imgsz_disp / 2)
    x0 = np.int16(w / 2 - imgsz_disp / 2)
    x1 = np.int16(w / 2 + imgsz_disp / 2)
    frame = frame[y0:y1, x0:x1, :]
    return frame


def getFdFovCropScale(frame, fd_box, imgsz_disp):
    """function to crop and resize image

    Args:
        frame (cv::Mat): image matrix
        fd_box (list): crop params
        imgsz_disp (int): desired image size

    Returns:
        cv::Mat: image matrix
    """
    frame = frame[fd_box[0]:fd_box[1], fd_box[2]:fd_box[3], :]
    frame = cv2.resize(frame, (imgsz_disp, imgsz_disp),
                       interpolation=cv2.INTER_AREA)
    return frame


def calibrationFdFov(vs, config):
    """function to find the filed of view for the fact within the whole image

    Args:
        vs (USBCamVideoStream): _description_
        config (dict): dict of configurations
    """
    drowsinessAlert_obj0 = drowsinessAlert(config)

    # (startY, endY, startX, endX)
    fd_box_arr = np.zeros((config['AVG_WIN_FDFOV'], 4))
    for count in range(config['AVG_WIN_FDFOV']):
        frame = vs.read()
        frame = get_cropped_frame(frame, config['imgsz_disp'])
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
    fdWidth = config['FovFdRatio'] * (fd_box_mean[2] - fd_box_mean[0])
    fdHeight = config['FovFdRatio'] * (fd_box_mean[3] - fd_box_mean[1])
    fdFovSize = np.maximum(fdWidth, fdHeight)
    startXFdFov = np.maximum(0, fdXCentre - fdFovSize / 2)
    endXFdFov = np.minimum(config['imgsz_disp'], fdXCentre + fdFovSize / 2)
    startYFdFov = np.maximum(0, fdYCentre - fdFovSize / 2)
    endYFdFov = np.minimum(config['imgsz_disp'], fdYCentre + fdFovSize / 2)
    fdFov = np.array(
        [startXFdFov, startYFdFov, endXFdFov, endYFdFov], dtype='int')
    np.save('./lib/fdFovSize', fdFov)
    time.sleep(0.5)


def calibrationEarYawOffset(vs, config):
    """This function does the calibration scan to find the eye aspect ratio and viewing angle offset

    Args:
        vs (USBCamVideoStream): _description_
        config (dict): dict of configurations
    """
    drowsinessAlert_obj0 = drowsinessAlert(config)

    ear_calib_arr = np.zeros((config['AVG_WIN'], 1))
    angle_offset_arr = np.zeros((config['AVG_WIN'], 1))
    for count in range(config['AVG_WIN']):
        frame = vs.read()
        frame = get_cropped_frame(frame, config['imgsz_disp'])
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


def calibration(config):
    """This function does the calibration scan to find the face detection field of view, Eye aspect ratio threshold and viewing angle offset

    Args:
        config (dict): dict of configuration parameters
    """
    # ----- USB cam setup ---- #
    vs = USBCamVideoStream(resolution=(
        config['imgsz_disp'], config['imgsz_disp']), port=0).start()
    time.sleep(2.0)

    start_flag = False
    while True:
        frame = vs.read()
        frame = get_cropped_frame(frame, config['imgsz_disp'])

        if cv2.waitKey(1) & 0xFF == ord('s'):
            start_flag = True

        if start_flag:
            calibrationFdFov(vs, config)
            calibrationEarYawOffset(vs, config)
            start_flag = False
        else:
            cv2.putText(frame, 'Press s to start calibration', (30, 40), cv2.FONT_ITALIC, 1.0,
                        (255, 0, 0), 1)
            cv2.putText(frame, 'Press q to quit', (30, 100),
                        cv2.FONT_ITALIC, 1.0, (255, 0, 0), 1)

        cv2.imshow("calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.stop()
    cv2.destroyAllWindows()


def run_single_thread(config):
    """function to run face detection CNN, face landmark detection and face pose estimation

    Args:
        config (dict): dict of configuration
    """
    # --- Initialize video writer object ---- #
    framesBuffer = deque(maxlen=150)
    delTime = 2.0
    # ---- Instantiate detection objects and Initialize  threads for the first time ---- #
    drowsinessAlert_obj = drowsinessAlert(config)

    # ----- Eye closure detection ----- #
    ear_obj = eye_closure_detection(config)
    # ----- Distraction detection ---- #
    if config['DISTR_DET']:
        distraction_obj = distraction_detection(buff_size=config['buff_size'], angle_thrshld_max=config['angle_thrshld_max'],
                                                angle_thrshld_min=config['angle_thrshld_min'], DEBUG=config['DEBUG'])
    # ----- Initilize variables ---- #
    start_fps, timeEyeclose = time.time(), time.time() - delTime
    count, fps, flagWrite = 0, 0, False,

    # ----- USB cam setup ---- #
    vs = USBCamVideoStream(resolution=(
        config['imgsz_disp'], config['imgsz_disp']), port=0).start()
    time.sleep(2.0)
    while True:
        frame_pi = vs.read()
        frame_pi = get_cropped_frame(frame_pi, config['imgsz_disp'])
        # if isAdjustFdFov:
        #     frame_pi = getFdFovCropScale(frame_pi, fd_box, imgsz_disp)
        frame_out, ear, euler_angle = drowsinessAlert_obj(frame_pi)
        # ----- FPS computation ----- #
        count += 1
        if count == config['NUM']:
            end_fps = time.time()
            elapsed_time = end_fps - start_fps
            start_fps = time.time()
            fps = np.ceil(config['NUM'] / elapsed_time)
            count = 0

        cv2.putText(frame_out, 'FPS: ' + str(fps), (300, 40),
                    cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
        # ---- Drowsiness Detection ---- #
        frame_out, flagEyeClose = ear_obj(frame_out, ear)
        if flagEyeClose and (time.time() - timeEyeclose) > delTime:
            timeEyeclose = time.time()
            flagWrite = True

        # ---- store frames buffer and write to file --- #
        if config['SAVE_VID']:
            framesBuffer.append(frame_out)
            if flagWrite and (time.time() - timeEyeclose) > delTime:
                flagWrite = False
                Thread(target=SaveVideoBuffer, args=(
                    './savedVideo/unsend/' + datetime.now().strftime("Date-%b:%d:%Y_Time-%H.%M.%S") + '.mp4',
                    framesBuffer.copy(
                    ), (config['imgsz_disp'], config['imgsz_disp']),
                    25.0)).start()

        # --- Distraction Detection ---- #
        if config['DISTR_DET']:
            distraction_obj(euler_angle[1, 0])
        if config['SHOW_IMG']:
            cv2.imshow("DSMS", frame_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vs.stop()
                break
        else:
            print('FPS:', fps)


def run_multi_thread(config):
    """function to run face detection CNN, face landmark detection and face pose estimation with multithreading

    Args:
        config (dict): dict of configuration
    """
    # --- Initialize video writer object ---- #
    framesBuffer = deque(maxlen=150)
    delTime = 2.0

    # ----- Drowsiness detection ----- #
    ear_obj = eye_closure_detection(config)
    euler_angle = np.zeros((3, 1))
    # ----- Distraction detection ---- #
    if config['DISTR_DET']:
        distraction_obj = distraction_detection(buff_size=config['buff_size'], angle_thrshld_max=config['angle_thrshld_max'],
                                                angle_thrshld_min=config['angle_thrshld_min'], DEBUG=config['DEBUG'])
    # ----- USB cam setup ---- #
    vs = USBCamVideoStream(resolution=(
        config['imgsz_disp'], config['imgsz_disp']), port=0).start()
    time.sleep(2.0)
    # ---- Instantiate detection objects and Initialize  threads for the first time ---- #
    if config['NUM_PROC'] >= 1:
        drowsinessAlert_obj_0 = drowsinessAlert(config)
        pool_0 = ThreadPool(processes=1)
        frame_0 = vs.read()
        frame_0 = get_cropped_frame(frame_0, config['imgsz_disp'])
        results_0 = pool_0.apply_async(drowsinessAlert_obj_0, args=(frame_0,))
    if config['NUM_PROC'] >= 2:
        drowsinessAlert_obj_1 = drowsinessAlert(config)
        pool_1 = ThreadPool(processes=1)
        frame_1 = vs.read()
        frame_1 = get_cropped_frame(frame_1, config['imgsz_disp'])
        results_1 = pool_1.apply_async(drowsinessAlert_obj_1, args=(frame_1,))
    if config['NUM_PROC'] >= 3:
        drowsinessAlert_obj_2 = drowsinessAlert(config)
        pool_2 = ThreadPool(processes=1)
        frame_2 = vs.read()
        frame_2 = get_cropped_frame(frame_2, config['imgsz_disp'])
        results_2 = pool_2.apply_async(drowsinessAlert_obj_2, args=(frame_2,))
    if config['NUM_PROC'] >= 4:
        drowsinessAlert_obj_3 = drowsinessAlert(config)
        pool_3 = ThreadPool(processes=1)
        frame_3 = vs.read()
        frame_3 = get_cropped_frame(frame_3, config['imgsz_disp'])
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
            frame_0 = get_cropped_frame(frame_0, config['imgsz_disp'])
            results_0 = pool_0.apply_async(
                drowsinessAlert_obj_0, args=(frame_0,))
        if indx == 1:
            output = results_1.get()
            frame_out = output[0]
            ear = output[1]
            euler_angle = output[2]
            frame_1 = vs.read()
            frame_1 = get_cropped_frame(frame_1, config['imgsz_disp'])
            results_1 = pool_1.apply_async(
                drowsinessAlert_obj_1, args=(frame_1,))
        if indx == 2:
            output = results_2.get()
            frame_out = output[0]
            ear = output[1]
            euler_angle = output[2]
            frame_2 = vs.read()
            frame_2 = get_cropped_frame(frame_2, config['imgsz_disp'])
            results_2 = pool_2.apply_async(
                drowsinessAlert_obj_2, args=(frame_2,))
        if indx == 3:
            output = results_3.get()
            frame_out = output[0]
            ear = output[1]
            euler_angle = output[2]
            frame_3 = vs.read()
            frame_3 = get_cropped_frame(frame_3, config['imgsz_disp'])
            results_3 = pool_3.apply_async(
                drowsinessAlert_obj_3, args=(frame_3,))

        count += 1
        indx += 1
        if indx == config['NUM_PROC']:
            indx = 0
        # ----- FPS computation ----- #
        if count == config['NUM']:
            end_fps = time.time()
            elapsed_time = end_fps - start_fps
            fps = np.ceil(config['NUM'] / elapsed_time)
            count = 0

        cv2.putText(frame_out, 'FPS: ' + str(fps), (300, 40),
                    cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
        # ---- Drowsiness Detection ---- #
        frame_out, flagEyeClose = ear_obj(frame_out, ear)
        if flagEyeClose and (time.time() - timeEyeclose) > delTime:
            timeEyeclose = time.time()
            flagWrite = True

        # ---- store frames buffer and write to file --- #
        if config['SAVE_VID']:
            framesBuffer.append(frame_out)
            if flagWrite and (time.time() - timeEyeclose) > delTime:
                flagWrite = False
                Thread(target=SaveVideoBuffer, args=(
                    './savedVideo/unsend/' + datetime.now().strftime("Date-%b:%d:%Y_Time-%H.%M.%S") + '.mp4',
                    framesBuffer.copy(
                    ), (config['imgsz_disp'], config['imgsz_disp']),
                    25.0)).start()

        # --- Distraction Detection ---- #
        if config['DISTR_DET']:
            distraction_obj(euler_angle[1, 0])
        if config['SHOW_IMG']:
            cv2.imshow("DSMS", frame_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vs.stop()
                break
        else:
            print('FPS: ', fps)


def run(config):
    """function to run DSMS core loop with option to use multiprocessing

    Args:
        config (dict): dict of configuration parameters
    """

    if config['MULTIPROC_FLAG']:
        run_multi_thread(config)
    else:
        run_single_thread(config)
