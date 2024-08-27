import cv2
import numpy as np

# Define camera IDs
CamL_id = 1  # Left camera ID
CamR_id = 0  # Right camera ID

# Initialize cameras
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

# Check if cameras opened successfully
if not CamL.isOpened() or not CamR.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# StereoSGBM parameters
minDisparity = 0
numDisparities = 64
blockSize = 8
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 8

# Create StereoSGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    disp12MaxDiff=disp12MaxDiff,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange
)

# Define the focal length and baseline
focal_length = 0.2142356409  # Example focal length in meters, replace with your value
baseline = 0.0006  # Example baseline in meters, replace with your value

while True:
    retL, frameL = CamL.read()
    retR, frameR = CamR.read()
    
    if not retL or not retR:
        print("Error: Unable to capture frames.")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity map
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    
    # Convert disparity to depth
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    non_zero_disparity = disparity > 0  # Avoid division by zero
    depth_map[non_zero_disparity] = (focal_length * baseline) / disparity[non_zero_disparity]
    
    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    # Apply a colormap for better visualization
    depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    
    # Display the depth map
    cv2.imshow('Depth Map', depth_map_color)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and close windows
CamL.release()
CamR.release()
cv2.destroyAllWindows()
