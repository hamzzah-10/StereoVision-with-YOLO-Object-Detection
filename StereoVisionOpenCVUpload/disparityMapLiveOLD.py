import cv2
import numpy as np

# Load saved calibration results and rectification maps
mtxL = np.load(r'parameters\mtxL.npy')
distL = np.load(r'parameters\distL.npy')
mtxR = np.load(r'parameters\mtxR.npy')
distR = np.load(r'parameters\distR.npy')
R = np.load(r'parameters\stereo_R.npy')
T = np.load(r'parameters\stereo_T.npy')
map1x = np.load(r'parameters\map1x.npy')
map1y = np.load(r'parameters\map1y.npy')
map2x = np.load(r'parameters\map2x.npy')
map2y = np.load(r'parameters\map2y.npy')

# Initialize cameras
CamL = cv2.VideoCapture(1)  # Update the camera ID if necessary
CamR = cv2.VideoCapture(0)  # Update the camera ID if necessary

# Check if cameras opened successfully
if not CamL.isOpened() or not CamR.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# StereoSGBM parameters
blockSize = 5
numDisparities = 32
P1 = 8 * 3 * blockSize ** 2
P2 = 32 * 3 * blockSize ** 2
disp12MaxDiff = 2
preFilterCap = 100
uniquenessRatio = 3
speckleWindowSize = 100
speckleRange = 32

# Create StereoSGBM object
stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange
    )

while True:
    # Capture frames
    retL, frameL = CamL.read()
    retR, frameR = CamR.read()
    
    if not retL or not retR:
        print("Error: Unable to capture frames.")
        break
    
    # Apply rectification
    rectifiedL = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

    # Convert to grayscale
    grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    grayR = cv2.GaussianBlur(grayR, (5, 5), 0)
    
    # Compute disparity map
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    
    # Handle disparity values that are zero
    disparity[disparity == 0] = np.nan
    
    # Normalize disparity map for visualization
    disparity_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)
    
    # Display results
    cv2.imshow('Rectified Left', grayL)
    cv2.imshow('Rectified Right', grayR)
    cv2.imshow('Disparity Map', disparity_display)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and close windows
CamL.release()
CamR.release()
cv2.destroyAllWindows()
