import mediapipe as mp
import cv2
import time




# Gesture recognition imports
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand landmark imports
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video = cv2.VideoCapture(0)


def display_frame(frame):
    cv2.imshow('Camera Feed', frame)
    cv2.waitKey(1)


def clear_frame(frame):


    pass


def overlay_image(image_path, target_region, gesture_name):
    a = cv2.imread(image_path)
    a = cv2.resize(a, (200, 200))
    frame[target_region[0]:target_region[1], target_region[2]:target_region[3]] = a

        # Print the gesture name below the image
    cv2.putText(frame, gesture_name, (target_region[2] + 10, target_region[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Assuming 'result.gestures' is a list
    if result.gestures:
        gesture_name = result.gestures[0][0].category_name
        print(gesture_name)

        if gesture_name == "adi mudra":
            image_path = 'C:/Users/vijay/OneDrive/Desktop/Gesture reconization/adi mudra/Adi-Mudra.jpeg'
            overlay_image(image_path, (0, 200, 0, 200), gesture_name)

        elif gesture_name == "vayu mudra":
            image_path = 'C:/Users/vijay/OneDrive/Desktop/Gesture reconization/vayu mudra/images (1).jpg'
            overlay_image(image_path, (0, 200, 0, 200), gesture_name)

        elif gesture_name == "apan mundra":  
            image_path = 'C:/Users/vijay/OneDrive/Desktop/Gesture reconization/apan mundra/images (10).jpg'
            overlay_image(image_path, (0, 200, 0, 200), gesture_name)









options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="D:/Downloads/gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer, mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while video.isOpened():
        ret, frame = video.read()


        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, timestamp)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    print(f"Landmark {landmark_id}: ({x}, {y})")


        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

video.release()
cv2.destroyAllWindows()