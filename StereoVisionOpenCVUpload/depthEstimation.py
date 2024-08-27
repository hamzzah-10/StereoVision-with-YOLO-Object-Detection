import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import time

# Load YOLOv8 model
model = YOLO(r'ourBest.pt')

# Define camera IDs
CamL_id = 0 # left camera
CamR_id = 1 # right camera

# Initialize cameras
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

# Check if cameras opened successfully
if not CamL.isOpened() or not CamR.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Load camera calibration results
mtxL = np.load(r'parameters\mtxL.npy')
distL = np.load(r'parameters\distL.npy')
mtxR = np.load(r'parameters\mtxR.npy')
distR = np.load(r'parameters\distR.npy')

# Load stereo rectification parameters
R1 = np.load(r'parameters\R1.npy')
R2 = np.load(r'parameters\R2.npy')
P1 = np.load(r'parameters\P1.npy')
P2 = np.load(r'parameters\P2.npy')
Q = np.load(r'parameters\Q.npy')

# Extract focal length from projection matrix P1
focal_length = P1[0, 0]

# Define the baseline
baseline = 0.102

# Initialize rectification maps
map1x, map1y = np.load(r'parameters\map1x.npy'), np.load(r'parameters\map1y.npy')
map2x, map2y = np.load(r'parameters\map2x.npy'), np.load(r'parameters\map2y.npy')

# Create StereoSGBM object with adjusted parameters
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=7,
    P1=8 * 3 * 7 ** 2,
    P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)

# Set confidence threshold
confidence_threshold = 0.6

# Scaling factor to correct depth estimation
scaling_factor = 0.94

# Conversion factor for meters to steps
meters_to_steps_conversion = 2  # 2 steps per meter

# List of class names
names = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench',
         'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat',
         'cell phone', 'chair', 'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant',
         'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
         'laptop', 'microwave', 'motorbike', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza',
         'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis',
         'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear',
         'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tvmonitor',
         'umbrella', 'vase', 'wine glass', 'zebra', 'stair step', 'door', 'tree', 'wall']

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Lock for synchronizing TTS
tts_lock = threading.Lock()

# Variables for real-time detection and delay
latest_summary = ""
summary_update_interval = 15  # seconds
last_summary_time = time.time()

def speak_summary(summary_message):
    with tts_lock:
        engine.say("Detected: " + summary_message)
        engine.runAndWait()

def summarize_and_speak():
    global latest_summary, last_summary_time
    while True:
        current_time = time.time()
        if current_time - last_summary_time >= summary_update_interval:
            speak_summary(latest_summary)
            last_summary_time = current_time
        time.sleep(1)

# Start the TTS summary thread
tts_thread = threading.Thread(target=summarize_and_speak, daemon=True)
tts_thread.start()

while True:
    retL, frameL = CamL.read()
    retR, frameR = CamR.read()
    
    if not retL or not retR:
        print("Error: Unable to capture frames.")
        break

    # Resize frames for faster processing
    frameL = cv2.resize(frameL, (640, 480))
    frameR = cv2.resize(frameR, (640, 480))

    # Rectify the images
    rectL = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    grayR = cv2.GaussianBlur(grayR, (5, 5), 0)
    
    # Apply bilateral filter to the rectified images
    filteredL = cv2.bilateralFilter(grayL, 9, 75, 75)
    filteredR = cv2.bilateralFilter(grayR, 9, 75, 75)
    
    # Compute disparity map
    disparity = stereo.compute(filteredL, filteredR).astype(np.float32) / 16.0

    # Create the WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    # Create a right matcher
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    # Compute disparity for the right image
    disparity_right = right_matcher.compute(grayR, grayL).astype(np.float32) / 16.0

    # Apply WLS filter
    filtered_disparity_map = wls_filter.filter(disparity, grayL, disparity_map_right=disparity_right)

    # Handle disparity values that are NaN
    filtered_disparity_map = np.nan_to_num(filtered_disparity_map, nan=0)

    # Normalize disparity map for visualization
    disparity_display = cv2.normalize(filtered_disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)

    # Apply median blur to the disparity map to reduce noise
    disparity_display = cv2.medianBlur(disparity_display, 5)

    # Convert disparity to depth with scaling factor
    depth_map = np.zeros_like(disparity_display, dtype=np.float32)
    depth_map[disparity_display > 0] = (focal_length * baseline * scaling_factor) / disparity_display[disparity_display > 0]
    
    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    # Apply a colormap for better visualization
    depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    
    # Perform object detection
    results = model(frameL)
    
    # Initialize counts and distances
    object_counts = {}
    object_distances = {}

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf < confidence_threshold:
                continue

            # Extract the bounding box coordinates and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color for the bounding box
            frameL = cv2.rectangle(frameL, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence
            label = f"{names[cls]}: {conf:.2f}"
            frameL = cv2.putText(frameL, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate the center of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Calculate disparity at the center of the bounding box
            disparity_at_center = filtered_disparity_map[cy, cx]
            
            # Calculate depth from disparity
            if disparity_at_center > 0:
                depth = (focal_length * baseline * scaling_factor) / disparity_at_center
            else:
                depth = 1.0  # default distance if disparity is zero

            # Convert depth to steps
            steps = int(depth * meters_to_steps_conversion)
            
            # Update object counts and distances
            object_name = names[cls]
            if object_name not in object_counts:
                object_counts[object_name] = 0
                object_distances[object_name] = []
            object_counts[object_name] += 1
            object_distances[object_name].append(depth)

            # Print object details with disparity and depth
            print(f"Object: {object_name} | Disparity: {disparity_at_center:.2f} | Depth: {depth:.2f} meters | Steps: {steps}")

    # Update the latest summary
    latest_summary = ", ".join(f"{count} {name}(s) at {int(np.mean(object_distances[name]) * meters_to_steps_conversion)} steps" for name, count in object_counts.items())

    # Display the updated frame with bounding boxes and labels
    cv2.imshow('Detected Objects', frameL)

    # Display disparity map
    cv2.imshow("Disparity Map", disparity_display)
    cv2.imshow("Depth Map", depth_map_color)

    # Display rectified frames
    #cv2.imshow("Rectified Left", rectL)
    #cv2.imshow("Rectified Right", rectR)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
CamL.release()
CamR.release()
cv2.destroyAllWindows()
