import cv2

# Open the default camera
cam = cv2.VideoCapture(2)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)
    

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("myimage.jpg", frame)
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()


moin = cv2.imread("myimage.jpg")
print(type(moin))