import cv2

def open_camera(camera_index=1):
    # Open the camera with the specified index
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Unable to open camera with index {camera_index}")
        return
    
    print(f"Camera with index {camera_index} opened successfully.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Unable to read frame from camera.")
            break
        
        # Display the resulting frame
        cv2.imshow(f'Camera {camera_index}', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    camera_index = 1  # Change this to the index of the camera you want to use
    open_camera(camera_index)

if __name__ == "__main__":
    main()
