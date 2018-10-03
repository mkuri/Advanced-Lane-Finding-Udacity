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


def window(img, n_windows=9, margin=100, minpix=50, f_img=None):
    width, height = img.shape[1], img.shape[0]
    if f_img == True:
        out_img = np.dstack((img, img, img))*255
    histogram = np.sum(img[height//2:,:], axis=0)
    midpoint = np.int(width/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(height/n_windows)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(n_windows):
        win_y_low = height - (window+1)*window_height
        win_y_high = height - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if f_img == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                    (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                    (0,255,0), 2) 
    
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)    
    left_fit_real = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_real = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)  

    if f_img == True:
        ploty = np.linspace(0, height-1, height)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')

        return(left_fit, right_fit, left_fit_real, right_fit_real, fig)
    
    return (left_fit, right_fit, left_fit_real, right_fit_real)

def save_window_imgs(img_paths, M, n_windows=9, margin=100, minpix=50):
    font = {'family': 'IPAexGothic',
            'color': 'black',
            'weight': 'normal',
            'size': 10,
            }

    for path in img_paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height = img.shape[1], img.shape[0]

        binary = generate_binary(img)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray')
        filename = './output_images/binary/' + path.stem + '_binarized.jpg'
        plt.savefig(fname=filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        warped = cv2.warpPerspective(binary, M, (width, height))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(warped, cmap='gray')
        filename = './output_images/warped/' + path.stem + '_warped.jpg'
        plt.savefig(fname=filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        left_fit, right_fit, left_fit_real, right_fit_real, fig = window(warped, f_img=True)

        filename = './output_images/window/' + path.stem + '_window.jpg'
        plt.savefig(fname=filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def pipeline(img, M):
    width, height = img.shape[1], img.shape[0]

    binary = generate_binary(img)
    warped = cv2.warpPerspective(binary, M, (width, height))
    


def main():
    # cam_calib_parameters = calibrate_camera(Path('./').glob('camera_cal/*.jpg'))
    # save_undist_imgs(Path('./').glob('camera_cal/*.jpg'), cam_calib_parameters)

    # save_binary_imgs(Path('./').glob('test_imgs/*.jpg'))

    # img = cv2.imread('./test_images/test1.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # M = get_transformation_matrix(img)
    # save_warped_imgs(Path('./').glob('test_images/*.jpg'), M)

    with open('./M.p', 'rb') as f:
        M = pickle.load(f)
    save_window_imgs(Path('./').glob('test_images/*.jpg'), M)




if __name__ == '__main__':
    main()
