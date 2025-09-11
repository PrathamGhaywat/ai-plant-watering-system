import cv2

# Open the default camera (usually the built-in webcam)
def take_picture(path_to_save: str ,port: int):
    cap = cv2.VideoCapture(port)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as an image
        cv2.imwrite(path_to_save, frame)
        print("Photo saved as " + path_to_save)
    else:
        print("Failed to capture image")

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()


