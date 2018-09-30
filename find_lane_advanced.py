#!/usr/bin/env python

from pathlib import Path
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io

import detect_edge


def calibrate_camera(img_paths):
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for path in img_paths:
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

    img_size = (img.shape[0], img.shape[1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    cam_calib_parameters = {}
    cam_calib_parameters['mtx'] = mtx
    cam_calib_parameters['dist'] = dist

    pickle.dump(cam_calib_parameters, open('./cam_calib_parameters.p', 'wb'))

    return cam_calib_parameters

def save_undist_imgs(img_paths, cam_calib_parameters):
    mtx = cam_calib_parameters['mtx']
    dist = cam_calib_parameters['dist']

    for path in img_paths:
        img = cv2.imread(str(path))
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('./camera_cal_undist/' + path.stem + '_undisttorted.jpg', undist)


def generate_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    sobelx_binary = detect_edge.abs_sobel_thresh(gray, orient='x', thresh=(20, 100))
    s_sobelx_binary = detect_edge.abs_sobel_thresh(s_channel, orient='x', thresh=(20, 100))

    combo_binary = np.zeros_like(s_channel)
    combo_binary[(sobelx_binary == 1) | (s_sobelx_binary == 1)] = 1

    return combo_binary


def save_binary_imgs(img_paths):
    for path in img_paths:
        img = skimage.io.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        sobelx_binary = detect_edge.abs_sobel_thresh(gray, orient='x', thresh=(20, 100))

        s_sobelx_binary = detect_edge.abs_sobel_thresh(s_channel, orient='x', thresh=(20, 100))

        combo_binary = np.zeros_like(s_channel)
        combo_binary[(sobelx_binary == 1) | (s_sobelx_binary == 1)] = 1

        font = {'family': 'IPAexGothic',
                'color': 'black',
                'weight': 'normal',
                'size': 10,
                }

        plt.subplot(221)
        plt.imshow(img)
        plt.title('Original', fontdict=font)
        plt.subplot(222)
        plt.imshow(sobelx_binary, cmap='gray')
        plt.title('X-gradient gray', fontdict=font)
        plt.subplot(223)
        plt.imshow(s_sobelx_binary, cmap='gray')
        plt.title('X-gradient hls s channel', fontdict=font)
        plt.subplot(224)
        plt.imshow(combo_binary, cmap='gray')
        plt.title('Combined', fontdict=font)

        filename = './binary_images/' + path.stem + '_binarized.jpg'
        plt.savefig(fname=filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)


def get_transformation_matrix(img):
    width, height = img.shape[1], img.shape[0]

    src = np.float32(
        [[(width / 2) - 55, height / 2 + 100],
        [((width / 6) - 10), height],
        [(width * 5 / 6) + 60, height],
        [(width / 2 + 55), height / 2 + 100]])
    dst = np.float32(
        [[(width / 4), 0],
        [(width / 4), height],
        [(width * 3 / 4), height],
        [(width * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (width, height))

    font = {'family': 'IPAexGothic',
            'color': 'black',
            'weight': 'normal',
            'size': 10,
            }
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original', fontdict=font)
    plt.subplot(122)
    plt.imshow(warped)
    plt.title('Warped', fontdict=font)

    plt.show()

    pickle.dump(M, open('./M.p', 'wb'))
    return M


def save_warped_imgs(img_paths, M, cmap=None):
    font = {'family': 'IPAexGothic',
            'color': 'black',
            'weight': 'normal',
            'size': 10,
            }

    for path in img_paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width = img.shape[1]
        height = img.shape[0]

        warped = cv2.warpPerspective(img, M, (width, height))

        plt.subplot(121)
        plt.imshow(img)
        plt.title('Original', fontdict=font)
        plt.subplot(122)
        plt.imshow(warped)
        plt.title('Warped', fontdict=font)

        filename = './warped_images/' + path.stem + '_warped.jpg'
        plt.savefig(fname=filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)


def pipeline(img, M):
    width, height = img.shape[1], img.shape[0]

    binary = generate_binary(img)
    warped = cv2.warpPerspective(binary, M, (width, height))
    


def main():
    # cam_calib_parameters = calibrate_camera(Path('./').glob('camera_cal/*.jpg'))
    # save_undist_imgs(Path('./').glob('camera_cal/*.jpg'), cam_calib_parameters)

    # save_binary_imgs(Path('./').glob('test_imgs/*.jpg'))

    img = cv2.imread('./test_images/test1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    M = get_transformation_matrix(img)
    save_warped_imgs(Path('./').glob('test_images/*.jpg'), M)




if __name__ == '__main__':
    main()
