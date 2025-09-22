# real_time_recognition_OPTIMIZED.py

import face_recognition
import pickle
import cv2

print("[INFO] Loading encodings...")
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

print("[INFO] Starting video stream...")
video_capture = cv2.VideoCapture(0)

# --- OPTIMIZATION VARIABLES ---
# 1. Resize frame for faster processing
RESIZE_FACTOR = 0.25  # Resize to 1/4 of the original size. Adjust as needed.

# 2. Skip frames to reduce workload
FRAME_SKIP = 5        # Process only every 5th frame.
frame_count = 0

# Variables to store the results from the last processed frame
last_known_locations = []
last_known_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # We will process the frame only if it's the Nth frame
    if process_this_frame:
        # --- HEAVY LIFTING IS DONE HERE ---
        # Reset last known results
        last_known_locations = []
        last_known_names = []

        # Resize the frame for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and compute encodings on the SMALL frame
        boxes = face_recognition.face_locations(rgb_small_frame, model="cnn") # or "hog"
        encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

        # Loop over the encodings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.55)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            
            last_known_names.append(name)
        
        last_known_locations = boxes
    
    # Toggle the flag for the next iteration
    process_this_frame = not process_this_frame

    # --- DRAWING IS DONE ON EVERY FRAME (for smooth video) ---
    # Use the results from the last processed frame
    for (top, right, bottom, left), name in zip(last_known_locations, last_known_names):
        # Scale back up face locations since the frame we detected in was resized
        top = int(top / RESIZE_FACTOR)
        right = int(right / RESIZE_FACTOR)
        bottom = int(bottom / RESIZE_FACTOR)
        left = int(left / RESIZE_FACTOR)

        # Draw a box around the face on the ORIGINAL frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Office Floor Monitoring", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("[INFO] Cleaning up...")
video_capture.release()
cv2.destroyAllWindows()