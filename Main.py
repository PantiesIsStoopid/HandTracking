import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

def count_fingers(landmarks):
    fingers_up = 0

    # Thumb: Check if the tip (landmark 4) is above the base (landmark 2)
    # Also check if the tip is horizontally to the right of the base (for right hand)
    if landmarks[4].y < landmarks[2].y and landmarks[4].x > landmarks[3].x:
        fingers_up += 1

    # Index: If the tip (landmark 8) is above the middle joint (landmark 7), count as up
    if landmarks[8].y < landmarks[6].y:
        fingers_up += 1

    # Middle: If the tip (landmark 12) is above the middle joint (landmark 11), count as up
    if landmarks[12].y < landmarks[10].y:
        fingers_up += 1

    # Ring: If the tip (landmark 16) is above the middle joint (landmark 15), count as up
    if landmarks[16].y < landmarks[14].y:
        fingers_up += 1

    # Pinky: If the tip (landmark 20) is above the middle joint (landmark 19), count as up
    if landmarks[20].y < landmarks[18].y:
        fingers_up += 1

    return fingers_up

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If hands are detected, count fingers for each hand
    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            fingers_up = count_fingers(hand_landmarks.landmark)
            
            # Display the finger count for each hand
            hand_label = f"Hand {hand_index + 1} Fingers Up: {fingers_up}"
            cv2.putText(frame, hand_label, (10, 50 + hand_index * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw landmarks and connections
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
