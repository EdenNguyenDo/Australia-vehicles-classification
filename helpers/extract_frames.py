import os
import cv2

# Path to the input video
video_path = '../1_21.626.mp4'
output_folder = 'output_frames'

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}. Check the file path and format.")
    exit()


frame_count = 0

# Loop to read frames from the video
while True:
    ret, frame = cap.read()

    if not ret:
        break  # If no frame is read, exit the loop (end of video)

    # Save the frame as a JPG file
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted and saved to '{output_folder}'")
