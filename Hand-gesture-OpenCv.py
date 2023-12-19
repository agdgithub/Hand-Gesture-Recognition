import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import cv2

# Load the pre-trained CNN model
newmod = load_model(r"C:/Users/saura/Desktop/Hand-gesture-recognition-using-OpenCv-and-Cnn-master/Hand-gesture-recognition-using-OpenCv-and-Cnn-master\hand_gestures.h5")

# Initialize the background as None
background = None

# Accumulated weight for the running average background
accumulated_weight = 0.5

# ROI coordinates for hand gesture detection
roi_top, roi_bottom, roi_right, roi_left = 20, 300, 300, 600

# Function to calculate the accumulated average of the background
def calc_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

# Function to segment the hand region in the frame
def segment(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    else:
        # Find the contour with the maximum area
        hand_segment = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment

# Function to predict the hand gesture using the CNN model
def predict_gesture(img):
    width, height = 64, 64
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    test_img = tf.keras.preprocessing.image.img_to_array(resized)
    test_img = np.expand_dims(test_img, axis=0)
    result = newmod.predict(test_img)
    gesture_index = np.argmax(result)
    return gesture_index

# Open the camera
cam = cv2.VideoCapture(0)

# Initialize the frame count
num_frames = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Get the region of interest (hand) from the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Convert ROI to grayscale and apply blur for better contour detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # For the first 60 frames, calculate the accumulated average of the background
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        # Segment the hand region
        segmentation_result = segment(gray)

        # Check if hand segment is detected
        if segmentation_result is not None:
            hand, hand_segment = segmentation_result

            # Draw contours around the hand segment
            epsilon = 0.001 * cv2.arcLength(hand_segment, True)
            hand_segment = cv2.approxPolyDP(hand_segment, epsilon, True)
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 2)

            # Predict the gesture using the CNN model
            gesture_index = predict_gesture(hand)
            gestures = ["Fist", "Five", "None", "Okay", "Peace", "Rad", "Straight", "Thumbs"]
            gesture_text = gestures[gesture_index]

            # Display the recognized gesture
            cv2.putText(frame_copy, gesture_text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw ROI rectangle on the frame
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)

    # Increment the frame count
    num_frames += 1

    # Display the frame with segmented hand and recognized gesture
    cv2.imshow("Hand Gesture Recognition", frame_copy)

    # Break the loop when 'Esc' key is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()