import cv2
import numpy as np
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objpoints = []  # 3d points in real world space
imgpointsL = []  # 2d points in image plane for left camera
imgpointsR = []  # 2d points in image plane for right camera

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Load images from directories
images_left = glob.glob('./images/StereoLeft/*.png')
images_right = glob.glob('./images/StereoRight/*.png')

# Detect corners in left and right images
for fnameL, fnameR in zip(images_left, images_right):
    imgL = cv2.imread(fnameL)
    imgR = cv2.imread(fnameR)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:
        objpoints.append(objp)
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsL.append(corners2L)
        imgpointsR.append(corners2R)

        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, CHECKERBOARD, corners2L, retL)
        cv2.drawChessboardCorners(imgR, CHECKERBOARD, corners2R, retR)

        # Visualize the images with detected corners
        cv2.imshow('Left Image with Corners', imgL)
        cv2.imshow('Right Image with Corners', imgR)
        cv2.waitKey(500)  # Display each image for 500 ms

cv2.destroyAllWindows()

# Proceed with calibration after displaying images
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# Save individual calibration results
np.save(r'parameters\mtxL.npy', mtxL)
np.save(r'parameters\distL.npy', distL)
np.save(r'parameters\rvecsL.npy', rvecsL)
np.save(r'parameters\tvecsL.npy', tvecsL)

np.save(r'parameters\mtxR.npy', mtxR)
np.save(r'parameters\distR.npy', distR)
np.save(r'parameters\rvecsR.npy', rvecsR)
np.save(r'parameters\tvecsR.npy', tvecsR)

print("Left camera matrix:\n", mtxL)
print("Left camera distortion coefficients:\n", distL)
print("Right camera matrix:\n", mtxR)
print("Right camera distortion coefficients:\n", distR)

# Compute reprojection error for each camera
def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error / len(object_points)

errorL = compute_reprojection_error(objpoints, imgpointsL, rvecsL, tvecsL, mtxL, distL)
errorR = compute_reprojection_error(objpoints, imgpointsR, rvecsR, tvecsR, mtxR, distR)

print(f"Left camera reprojection error: {errorL}")
print(f"Right camera reprojection error: {errorR}")

########################### Stereo Camera Calibration ####################################
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)

retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=criteria_stereo, flags=flags)

print("Stereo calibration done.")
print("Rotation matrix between the cameras: \n", R)
print("Translation vector between the cameras: \n", T)

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T)

# Compute the rectification maps
map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_32FC1)

# Save the rectification maps and parameters for future use
np.save(r'parameters\R1.npy', R1)
np.save(r'parameters\R2.npy', R2)
np.save(r'parameters\P1.npy', P1)
np.save(r'parameters\P2.npy', P2)
np.save(r'parameters\Q.npy', Q)
np.save(r'parameters\map1x.npy', map1x)
np.save(r'parameters\map1y.npy', map1y)
np.save(r'parameters\map2x.npy', map2x)
np.save(r'parameters\map2y.npy', map2y)

np.save(r'parameters\stereo_R.npy', R)
np.save(r'parameters\stereo_T.npy', T)
np.save(r'parameters\stereo_E.npy', E)
np.save(r'parameters\stereo_F.npy', F)

print("Calibration and rectification parameters have been saved.")

# RMS error for stereo calibration
print(f"Stereo calibration RMS error: {retS}")
