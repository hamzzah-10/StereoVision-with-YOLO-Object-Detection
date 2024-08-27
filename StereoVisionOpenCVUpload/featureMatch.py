import cv2
import numpy as np

# Initialize the ORB detector
orb = cv2.ORB_create()

# Initialize the FLANN based matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, 
                    table_number=6,  
                    key_size=12,    
                    multi_probe_level=1)
search_params = dict()  # or set search params like {'checks': 50}

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Capture video from stereo cameras
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

while True:
    # Read frames from both cameras
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Error: Unable to capture frames.")
        break

    # Convert to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    kpL, desL = orb.detectAndCompute(grayL, None)
    kpR, desR = orb.detectAndCompute(grayR, None)

    # Ensure descriptors are not None and have enough points
    if desL is not None and desR is not None and desL.shape[0] > 0 and desR.shape[0] > 0:
        # Ensure descriptors are continuous
        desL = desL.copy()  # Ensure continuity
        desR = desR.copy()  # Ensure continuity

        # Match descriptors
        try:
            matches = flann.knnMatch(desL, desR, k=2)

            # Check if matches are not empty and unpack them correctly
            if matches:
                good_matches = []
                for match in matches:
                    if len(match) == 2:  # Ensure there are 2 matches for each descriptor
                        m, n = match
                        # Apply ratio test
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    else:
                        print(f"Unexpected match format: {match}")

                # Draw matches
                result = cv2.drawMatches(frameL, kpL, frameR, kpR, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                # Display results
                cv2.imshow('Feature Matches', result)
            else:
                print("No matches found.")

        except cv2.error as e:
            print(f"Error during FLANN matching: {e}")

    else:
        print("No valid descriptors found in one or both images.")

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capL.release()
capR.release()
cv2.destroyAllWindows()
