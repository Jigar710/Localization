import cv2
import os
import time
import pandas as pd
import os
from geopy.geocoders import Nominatim
import geocoder
import pywifi
from pywifi import const
import time
import requests


def collectData(room_name, room_type, building):
    if not os.path.exists('captured_images/'+room_name):
        os.mkdir('captured_images/'+room_name)
    
    cap = cv2.VideoCapture(0)  # 0 represents the default camera, you can change it if you have multiple cameras

# Counter for the captured images
    image_count = 0

# Create a loop to capture and save 20 images with a 1-second delay
    while image_count < 20:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Failed to grab frame")
            break

    # Save the captured frame to the folder
        image_filename = f'captured_images/'+ room_name +'/image'+str(image_count)+'.jpg'
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

    elevation_api_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lng}"
    response = requests.get(elevation_api_url)
    
    if response.status_code == 200:
        elevation_data = response.json()
        print(elevation_data)
        height = elevation_data["results"][0]["elevation"]
        print(f"Elevation: {height} meters")
    else:
        height = 0
        print("Failed to retrieve elevation data.")

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

    folder_path = "/"+room_name

    new_row = {"space":room_name,
               "latitude":lat,
               "longtitude":lng,
                "elevation" : height,
               "w01":w01,
               "w02":w02,
               "w03":w03,
               "w04":w04,
               "folderPath":folder_path,
               "spaceType": room_type,
               "building" : building,
               }

    
    # Specify the CSV file path
    csv_file_path = "spaceData.csv"

    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)

    # Create a DataFrame from the CSV file or an empty one if it doesn't exist
    if file_exists:
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame(columns=["space","latitude","longtitude","elevation","w01","w02","w03","w04","folderPath","spaceType","building"])

    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Row added to {csv_file_path}")

# collectData("D218","room","hostel")