import cv2
import ultralytics as uy
import requests as rq
import numpy as np
import base64
from fastapi import FastAPI
import dotenv


AI_API_KEY = "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" # openrouter api key
image_path = "./images/begonia.jpeg"
yolo = "./yolo11n.pt"
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
# Take Photo
# Load YOLO model 
model = uy.YOLO(yolo)
# Open the default camera
cam = cv2.VideoCapture(2)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


ret, frame = cam.read()
print("Please place the plant at a distance of 50cm. \n When you are done, remove all obstacles standing in the way and press 'q' to confirm ")
# Display the captured frame
cv2.imshow('Camera', frame)


# Press 'q' to exit the loop
if cv2.waitKey(1) == ord('q'):
    cv2.imwrite("./captures/plant_headshot1.jpg", frame)
        
done = input("Please turn the plant 90 degrees and type yes to confirm: ")
cv2.imwrite("./captures/plant_headshot2.jpg", frame)
# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
print("Done!")