import cv2
import ultralytics as uy
import requests as rq
import numpy as np
import base64
import serial
import json

AI_API_KEY = "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" # openrouter api key
image_path = "./images/begonia.jpeg"
plant_name = ""
identify_prompt = """What is in this image? eg if an apple is being shown then dont't describe it. 
just try to identify the apple and output the latin name only. 
no other description or text of it. 
also if you are only 100 percent sure, then only"""
water_consumption_prompt = f"""
Please give me the water consumption in ml. 
If for example a plant needs 400ml of waters the output will be: 400.
 Don't give me text or any type of description or say anything except the number. 
 If you fail to obey this you will be replaced by an better model. 
Only one number. Here is the name of the plant: {plant_name}
"""


"""
Setup Process. Will do this on the initial startup of the program.
"""
# Open the default camera
def take_picture(num: int):
    """
    Takes the picture of the plant. \n
    Input: Headshot number \n
    output: image written in ./captures folder
    """
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
            cv2.imwrite(f"./captures/plant_headshot{num}.jpg", frame)
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()

def yolo(image_path: str):
    """
    Runs YOLO object detection on the given image and calculates
    the detected object's dimensions relative to image size.
    """
    # Load a pretrained YOLO model
    model = uy.YOLO("yolo11n.pt")

    # Read the image
    image = cv2.imread(image_path)
    height, width, channels = image.shape  # type: ignore

    # Run inference
    results = model(image_path)

    # Show the first detection result
    print("📷 Displaying detection results for the image...")

    # --- Object Width ---
    print("\n--- Object Width Calculation ---")
    print("Bounding box (x1, y1, x2, y2):", results[0].boxes.xyxy)
    plant_width = abs(results[0].boxes.xyxy[0][0] - results[0].boxes.xyxy[0][2])
    print(f"Detected object width (pixels): {int(plant_width)}")
    print(f"Image width (pixels): {int(width)}")

    # --- Object Height ---
    print("\n--- Object Height Calculation ---")
    print("Bounding box (x1, y1, x2, y2):", results[0].boxes.xyxy)
    plant_height = abs(results[0].boxes.xyxy[0][1] - results[0].boxes.xyxy[0][3])

    # Calculate "real world" width (scaled values)
    print("\n--- Real-World Scale Approximations ---")
    real_width_44 = (float(plant_height) / float(width)) * 44

    real_width_46 = (float(plant_height) / float(width)) * 46
    return real_width_46, real_width_44

if __name__ == "__main__":
    
    """
    print(take_picture(1))
    ord("q")
    input("Did you turn the plant for 90 degrees? Once finished press enter and then prsess q to capture the plant")
    print(take_picture(2))
    ord("q")
    """
    bruh = yolo("./captures/plant_headshot1.jpg")
    print(bruh[0])
    #yolo("./captures/plant_headshot2.jpg")
