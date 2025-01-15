import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and face meshq
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Initialize previous mouse position for smoothing
prev_x, prev_y = screen_w / 2, screen_h / 2

def smooth_move(prev_x, prev_y, target_x, target_y, smooth_factor=0.2):
    # Smooth transition towards target with higher smooth_factor
    move_x = prev_x + (target_x - prev_x) * smooth_factor
    move_y = prev_y + (target_y - prev_y) * smooth_factor
    return move_x, move_y

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark
        frame_h, frame_w, _ = frame.shape

        # Loop through the selected facial landmarks
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:
                # Calculate the screen position
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y

                # Smooth the mouse movement with higher smoothing
                smooth_x, smooth_y = smooth_move(prev_x, prev_y, screen_x, screen_y)

                # Move the mouse to the smoothed position with a smaller movement rate
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)  # Reduce duration to make it faster
                prev_x, prev_y = smooth_x, smooth_y  # Update the previous position

        # Check for eye gesture for clicking
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()  # Perform the click action immediately without delay

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)
