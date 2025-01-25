import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
MpHands = mp.solutions.hands
Hands = MpHands.Hands()

# Initialize OpenCV video capture
Cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    Ret, Frame = Cap.read()

    if not Ret:
        break

    # Flip the frame horizontally for selfie-view
    Frame = cv2.flip(Frame, 1)

    # Convert BGR to RGB
    RgbFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    Results = Hands.process(RgbFrame)

    # If hands are detected, draw landmarks and connections
    if Results.multi_hand_landmarks:
        for HandLandmarks in Results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(Frame, HandLandmarks, MpHands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Tracking", Frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
Cap.release()
cv2.destroyAllWindows()


