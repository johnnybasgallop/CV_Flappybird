import math

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Drawing specs for circles
circles_color = mp_drawing.DrawingSpec(
    color=(255, 255, 255), thickness=10, circle_radius=2)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmarks for thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate Euclidean distance
                distance = math.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

                # Convert normalized coordinates to pixel coordinates
                image_height, image_width, _ = image.shape
                thumb_px = (int(thumb_tip.x * image_width),
                            int(thumb_tip.y * image_height))
                index_px = (int(index_tip.x * image_width),
                            int(index_tip.y * image_height))

                # Draw a line between the thumb tip and index finger tip
                # Blue line, thickness 2
                cv2.line(image, thumb_px, index_px, (255, 0, 0), 2)

                # Calculate midpoint for text placement
                midpoint_x = int((thumb_px[0] + index_px[0]) / 2)
                midpoint_y = int((thumb_px[1] + index_px[1]) / 2)

                # Display distance on screen above the midpoint
                cv2.putText(image, f"Distance: {distance:.2f}", (midpoint_x, midpoint_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10, cv2.LINE_AA)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    circles_color,
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
