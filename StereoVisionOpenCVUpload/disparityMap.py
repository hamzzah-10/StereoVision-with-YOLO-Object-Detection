import cv2
import numpy as np

# StereoSGBM parameters
blockSize = 5
numDisparities = 64
P1 = 8 * 3 * blockSize ** 2
P2 = 32 * 3 * blockSize ** 2
disp12MaxDiff = 2
preFilterCap = 45
uniquenessRatio = 10
speckleWindowSize = 100
speckleRange = 32

def compute_and_display_disparity():
    global blockSize, numDisparities, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange
    
    # Create StereoSGBM object
    stereoSGBM = cv2.StereoSGBM_create(
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

    # Apply rectification
    rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    
    # Convert to grayscale
    grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    grayR = cv2.GaussianBlur(grayR, (5, 5), 0)
    
    # Compute disparity map
    disparity_SGBM = stereoSGBM.compute(grayL, grayR).astype(np.float32)

    # Create the WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereoSGBM)
    wls_filter.setLambda(8000.0)  # You can adjust these parameters as needed
    wls_filter.setSigmaColor(1.5)

    # Create a right matcher
    right_matcher = cv2.ximgproc.createRightMatcher(stereoSGBM)

    # Compute disparity for the right image
    disparity_right = right_matcher.compute(grayR, grayL)

    # Apply WLS filter
    filtered_disparity_map = wls_filter.filter(disparity_SGBM, grayL, disparity_map_right=disparity_right)

    # Handle disparity values that are NaN
    filtered_disparity_map = np.nan_to_num(filtered_disparity_map, nan=0)

    # Normalize disparity map for visualization
    disparity_display = cv2.normalize(filtered_disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)
    
    # Apply median blur to the disparity map to reduce noise
    disparity_display = cv2.medianBlur(disparity_display, 5)
    
    # Display results
    cv2.imshow('Rectified Left', rectifiedL)
    cv2.imshow('Rectified Right', rectifiedR)
    cv2.imshow('Disparity Map', disparity_display)

if __name__ == "__main__":
    # Load calibration data and images
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

    imgL = cv2.imread(r'imageL0.png')
    imgR = cv2.imread(r'imageR0.png')
    
    if imgL is None or imgR is None:
        print("Error: One or both images could not be loaded.")
    else:
        compute_and_display_disparity()

        # Wait for user to press 'q' to exit
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
