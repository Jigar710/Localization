import cv2
import os
import time

# Create a folder to save the captured images
if not os.path.exists('captured_images'):
    os.mkdir('captured_images')

# Initialize the camera capture
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
    image_filename = f'captured_images/image{image_count}.jpg'
    cv2.imwrite(image_filename, frame)
    print(f"Image {image_count} saved as {image_filename}")

    image_count += 1

    # Wait for 1 second before capturing the next image
    time.sleep(1)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
