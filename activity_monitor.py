import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return math.sqrt(((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2) + ((p1.z - p2.z) ** 2))

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.setflags(write=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find poses
    results = pose.process(image)

    # Convert the image back to BGR
    image.flags.setflags(write=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Check for poses
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # --- ACTION RECOGNITION LOGIC ---
        activity = "Working / Idle" # Default state

        # Get coordinates for right hand and right ear
        # Landmark indices can be found on the MediaPipe website
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        
        # Calculate the distance
        # The visibility check ensures the landmarks are detected
        if right_wrist.visibility > 0.7 and right_ear.visibility > 0.7:
            distance = calculate_distance(right_wrist, right_ear)
            
            # This threshold is arbitrary and needs to be tuned.
            # A smaller distance implies the hand is near the ear.
            if distance < 0.15:
                activity = "On a Call"

        # Display the activity on the screen
        cv2.putText(image, f"Activity: {activity}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Productivity Monitor (Concept)', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()