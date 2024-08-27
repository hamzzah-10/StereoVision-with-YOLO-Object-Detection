import cv2
import time

# Open the cameras explicitly as left and right
cap_left = cv2.VideoCapture(0)  # Assuming camera 1 is the left camera
cap_right = cv2.VideoCapture(1)  # Assuming camera 0 is the right camera

num = 0
capture_interval = 4  # Interval in seconds
last_capture_time = time.time()

while cap_left.isOpened() and cap_right.isOpened():

    success_left, img_left = cap_left.read()
    success_right, img_right = cap_right.read()

    if not success_left or not success_right:
        print("Failed to capture images from one or both cameras.")
        break

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img_left)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img_right)
        print(f"Images saved! {num}")
        num += 1
        last_capture_time = current_time

    cv2.imshow('Left Camera', img_left)
    cv2.imshow('Right Camera', img_right)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
