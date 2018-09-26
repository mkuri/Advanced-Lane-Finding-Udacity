#!/usr/bin/env python

from pathlib import Path
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt


def calibrate_camera(image_paths):
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for path in image_paths:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('Calibration images', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    image_size = (img.shape[0], img.shape[1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    cam_calib_parameters = {}
    cam_calib_parameters['mtx'] = mtx
    cam_calib_parameters['dist'] = dist

    pickle.dump(cam_calib_parameters, open('./cam_calib_parameters.p', 'wb'))

    return cam_calib_parameters

def save_undist_images(image_paths, cam_calib_parameters):
    mtx = cam_calib_parameters['mtx']
    dist = cam_calib_parameters['dist']

    for path in image_paths:
        img = cv2.imread(str(path))
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('./camera_cal_undist/' + path.stem + '_undisttorted.jpg', undist)


def pipeline(f_cam_calib=False):

    if f_cam_calib == True:
        cam_calib_parameters = calibrate_camera(Path('./').glob('camera_cal/*.jpg'))
        # save_undist_images(Path('./').glob('camera_cal/*.jpg'), cam_calib_parameters)


if __name__ == '__main__':
    pipeline()
