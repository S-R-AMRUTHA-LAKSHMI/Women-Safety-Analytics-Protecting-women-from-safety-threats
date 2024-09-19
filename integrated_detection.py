import cv2
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from twilio.rest import Client
import mysql.connector
from fer import FER
from datetime import datetime
import cloudinary
import cloudinary.uploader

# Twilio credentials (replace with your actual credentials)
account_sid = 'ACac3b4400db9619ac36c7090041ac5f0e'
auth_token = 'f40fac9199de87569c29eafa6a5b1882'
twilio_phone_number = '+15203574586'

# Cloudinary configuration
cloudinary.config(
    cloud_name="dp6ep06xr",
    api_key="921498323232336",
    api_secret="y5LhuW9RGY0S76fkpd0SJLTJxCk"
)

# Load YOLOv5 for body detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load gender classification model
gender_model = load_model('gender_detection_body.keras')

# Load the violence detection model
violence_model = tf.keras.models.load_model('violence_detection_model.h5')

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Initialize the face detector for emotion analysis
face_detector = FER(mtcnn=True)

# Function to send SMS alert or make a call
def send_alert(message, mobile_number, is_call=False, screenshot_path=None):
    try:
        if screenshot_path:
            screenshot_url = upload_screenshot_to_cloudinary(screenshot_path)
            if not screenshot_url:
                print("Failed to upload screenshot. Aborting alert.")
                return

            # SMS message body with screenshot
            sms_message = client.messages.create(
                body=message,
                from_=twilio_phone_number,
                media_url=[screenshot_url],  # Use Cloudinary image URL
                to=mobile_number
            )
        else:
            sms_message = client.messages.create(
                body=message,
                from_=twilio_phone_number,
                to=mobile_number
            )
        print(f"SMS sent: {sms_message.body} to {mobile_number}")

        if is_call:
            twiml = f'<Response><Say>{message}</Say></Response>'
            call = client.calls.create(
                to=mobile_number,
                from_=twilio_phone_number,
                twiml=twiml
            )
            print(f"Call initiated: {message} to {mobile_number}")

    except Exception as e:
        print(f"Failed to send SMS or make call: {e}")

# Function to upload the screenshot to Cloudinary
def upload_screenshot_to_cloudinary(screenshot_path):
    try:
        upload_result = cloudinary.uploader.upload(screenshot_path)
        return upload_result['secure_url']  # Return the public URL
    except Exception as e:
        print(f"Error uploading screenshot to Cloudinary: {e}")
        return None

# Database connection function
def connect_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            port=3308,
            user="root",
            password="amrutha@2811",
            database="WomenSafetyDB"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Function to get the police station mobile number for a specific location
def get_police_mobile(cursor, location):
    try:
        query = "SELECT police_station_mobile FROM CameraData WHERE location = %s"
        cursor.execute(query, (location,))
        result = cursor.fetchone()
        return result[0] if result else None
    except mysql.connector.Error as err:
        print(f"Database query error: {err}")
        return None

# Function to increment alert count for a location
def increment_alert_count(conn, cursor, location):
    try:
        query = "UPDATE CameraData SET alert_count = alert_count + 1 WHERE location = %s"
        cursor.execute(query, (location,))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error incrementing alert count: {err}")

# Function to get the hotspot location
def get_hotspot_location(cursor):
    try:
        query = "SELECT location FROM CameraData ORDER BY alert_count DESC LIMIT 1"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result else None
    except mysql.connector.Error as err:
        print(f"Error getting hotspot location: {err}")
        return None

def preprocess_image(image):
    image_resized = cv2.resize(image, (96, 96))
    image_resized = image_resized.astype("float32") / 255.0
    return np.expand_dims(image_resized, axis=0)

def predict_gender(cropped_body_image):
    processed_image = preprocess_image(cropped_body_image)
    predictions = gender_model.predict(processed_image)
    gender = 'male' if np.argmax(predictions) == 0 else 'female'
    return gender

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    return normalized_frame



def analyze_video(video_path, location):
    conn = connect_db()
    if not conn:
        print("Failed to connect to the database.")
        return

    cursor = conn.cursor()
    police_mobile = get_police_mobile(cursor, location)

    video = cv2.VideoCapture(video_path)

    prev_lone_woman_detected = False
    prev_women_surrounded_by_men_detected = False
    danger_detected = False
    prev_danger_detected = False
    frame_sequence = []
    sequence_length = 30
    alert_sent = False
    screenshot_path = None

    while video.isOpened():
        status, frame = video.read()
        if not status:
            break

        # Process violence detection
        processed_frame = preprocess_frame(frame)
        frame_sequence.append(processed_frame)

        if len(frame_sequence) > sequence_length:
            frame_sequence.pop(0)

        if len(frame_sequence) == sequence_length:
            input_sequence = np.expand_dims(np.array(frame_sequence), axis=0)
            prediction = violence_model.predict(input_sequence)

            if prediction[0] > 0.7:
                if not prev_danger_detected:
                    danger_detected = True
                    prev_danger_detected = True

                    # Create a copy of the frame for the screenshot
                    screenshot_frame = frame.copy()
                    # Add "Danger Detected" text to the screenshot
                    cv2.putText(screenshot_frame, "Danger Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    # Save the screenshot with "Danger Detected" text
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, screenshot_frame)
                    print(f"Screenshot saved: {screenshot_path}")

                    # Send alert with screenshot and call
                    if police_mobile and not alert_sent:
                        send_alert(f"Danger detected at {location}. Immediate action required!", police_mobile, is_call=True, screenshot_path=screenshot_path)
                        increment_alert_count(conn, cursor, location)
                        alert_sent = True
            else:
                prev_danger_detected = False
                danger_detected = False
                alert_sent = False

        # Create a copy of the frame for display
        display_frame = frame.copy()

        if not danger_detected:
            # Process gender detection and add bounding boxes only if danger is not detected
            num_men = 0
            num_women = 0

            results = yolo_model(display_frame)
            detections = results.pandas().xyxy[0]

            for _, row in detections.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = row['name']

                if label == 'person':
                    body_crop = display_frame[y1:y2, x1:x2]
                    if body_crop.shape[0] > 0 and body_crop.shape[1] > 0:
                        gender = predict_gender(body_crop)
                        if gender == 'male':
                            num_men += 1
                        elif gender == 'female':
                            num_women += 1

                        color = (0, 255, 0) if gender == 'female' else (0, 0, 255)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f'Gender: {gender}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            lone_woman_detected = (num_women == 1 and num_men == 0)
            women_surrounded_by_men_detected = (num_women > 0 and num_men >= num_women)

            if lone_woman_detected and (not prev_lone_woman_detected) and police_mobile:
                send_alert(f"Alert: Lone Woman Detected at {location}", police_mobile)
            elif women_surrounded_by_men_detected and (not prev_women_surrounded_by_men_detected) and police_mobile:
                send_alert(f"Alert: Women Surrounded by Men Detected at {location}", police_mobile)

            prev_lone_woman_detected = lone_woman_detected
            prev_women_surrounded_by_men_detected = women_surrounded_by_men_detected

            if lone_woman_detected:
                cv2.putText(display_frame, "Lone Woman Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if women_surrounded_by_men_detected:
                cv2.putText(display_frame, "Woman Surrounded by Men Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            # If danger is detected, only add "Danger Detected" text to the display frame
            cv2.putText(display_frame, "Danger Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Video', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    cursor.close()
    conn.close()
def main():
    print("Starting main function...")
    conn = connect_db()
    if not conn:
        print("Failed to connect to the database.")
        return

    cursor = conn.cursor()

    # Get the hotspot location
    hotspot_location = get_hotspot_location(cursor)
    print(f"Current hotspot location: {hotspot_location}")

    # Prompt user for location
    location = input("Enter the location: ")

    # Check if the entered location is a hotspot
    if location == hotspot_location:
        police_mobile = get_police_mobile(cursor, location)
        if police_mobile:
            send_alert(f"SOS Alert! Assistance required at hotspot location {location}.", police_mobile)
            print(f"SOS alert sent for hotspot location: {location}")
        else:
            print("No police station mobile found for the hotspot location.")

    print("Starting video analysis...")
    # Analyze the video
    analyze_video('sample1.mp4', location)

    cursor.close()
    conn.close()
    print("Main function completed.")

if __name__ == "__main__":
    print("Script started.")
    main()
    print("Script ended.")