from ultralytics import YOLO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import os
import cv2
from geopy.geocoders import Nominatim
import geocoder
import pywifi
from pywifi import const
from keras.models import load_model
import joblib

def predict():
    if not os.path.exists('predict_images'):
        os.mkdir('predict_images')
    
    cap = cv2.VideoCapture(0)  # 0 represents the default camera, you can change it if you have multiple cameras

# Counter for the captured images
    image_count = 0

# Create a loop to capture and save 20 images with a 1-second delay
    while image_count < 2:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Failed to grab frame")
            break

    # Save the captured frame to the folder
        image_filename = f'predict_images/'+'/image'+str(image_count)+'.jpg'
        cv2.imwrite(image_filename, frame)
        print(f"Image {image_count} saved as {image_filename}")

        image_count += 1

    # Wait for 1 second before capturing the next image
        time.sleep(1)

    # Initialize the geocoder
    geolocator = Nominatim(user_agent="myGeocoder")

    # Use the geocoder library to get the current location
    current_location = geocoder.ip('me')

    if current_location:
        lat = current_location.latlng[0]
        lng = current_location.latlng[1]

        print(f"Current Latitude: {lat}")
        print(f"Current Longitude: {lng}")
    else:
        lat = 0
        lng = 0
        print("Failed to retrieve current location.")

    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]  # Assuming the first Wi-Fi interface

    # Scan for available networks
    iface.scan()

    # Get scan results
    scan_results = iface.scan_results()
    time.sleep(1)

    iface.scan()
    # Get scan results
    scan_results2 = iface.scan_results()

    scan_results.extend(scan_results2)
    freq = []
    wifilst = ["DAIICT_Student", "DA_Public", "Attendance", "DAIICT_STAFF"]
    for res in scan_results:
        if(res.ssid in wifilst):
            freq.append([res.ssid, res.signal])
    # Print the list of frequencies
    print("Wi-Fi Frequencies:")
    for frequency in freq:
        print(frequency)

    finallst = []
    w01,w02,w03,w04 = 0,0,0,0
    for i in wifilst:
        for j in freq:
            if(j[0]==i):
                if(j[0]==wifilst[0]):
                    w01 = j[1]
                if(j[0]==wifilst[1]):
                    w02 = j[1]
                if(j[0]==wifilst[2]):
                    w03 = j[1]
                if(j[0]==wifilst[3]):
                    w04 = j[1]
                finallst.append(j)
                break
    print(finallst)
    # Define the data for the new row

    folder_path = "/predict_images"
    oneRoomObj = set()
    yolo_model = YOLO('yolov8n.pt')
    for images in os.listdir("predict_images"):
        # check if the image ends with png
        if (images.endswith(".jpg")):
            print("predict_images/"+images)
            results = yolo_model.predict(source="predict_images/"+images,save=False)
            for i in range(len(results[0].boxes.cls)):
                oneRoomObj.add(results[0].names[int(results[0].boxes.cls[i])])
    new_row = {
               "latitude":lat,
               "longtitude":lng,
               "w01":w01,
               "w02":w02,
               "w03":w03,
               "w04":w04,
               "folderPath":folder_path
               }
    
    testdf = pd.DataFrame([new_row],columns=["latitude","longtitude","w01","w02","w03","w04","folderPath"])
    with open("set_as_list.txt", "r") as file:
        list_from_file = eval(file.read())
    objects = set(list_from_file)
    for obj in objects:
        if obj in oneRoomObj:
            testdf[str(obj)] = 1
        else:
            testdf[str(obj)] = 0

# Convert list to set
    
    # Append the new row to the DataFrame
    # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(testdf)
    model = load_model("model.h5")
    
    # space_encoder = joblib.load('space_encoder.pkl')
    # spaceT_encoder = joblib.load('spaceT_encoder.pkl')
    # building_encoder = joblib.load('building_encoder.pkl')

    # encoded_space  =  space_encoder.transform(df['space'])
    # df['space'] = encoded_space

    # encoded_spaceT  =  spaceT_encoder.transform(df['spaceType'])
    # df['spaceType'] = encoded_spaceT

    # encoded_building  =  building_encoder.transform(df['building'])
    # df['building'] = encoded_building

    # testdf = np.array(testdf)
    testdf = testdf.drop(['folderPath'],axis=1)
    # testdf = np.asarray(testdf).astype(np.float32)
    predictions = model.predict(testdf)
    print(predictions)
    y_pred = np.argmax(predictions, axis=-1)
    print(y_pred)

# predict()


    