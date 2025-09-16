# recognize_faces_fast.py
import face_recognition
import cv2
import os
import numpy as np
import pickle

# --- Configuration ---
UNKNOWN_FACES_DIR = "unknown_faces"
ENCODINGS_FILE = "encodings.pkl"
TOLERANCE = 0.6
MODEL = "hog"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

print("Loading saved encodings...")

# --- Load Known Faces from file ---
try:
    with open(ENCODINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    known_faces_encodings = data["encodings"]
    known_faces_names = data["names"]
except FileNotFoundError:
    print(f"Error: Encoding file '{ENCODINGS_FILE}' not found.")
    print("Please run the 'encode_faces.py' script first to generate the encodings.")
    exit()

print(f"Loaded {len(known_faces_names)} known faces.")

# --- Process Unknown Faces ---
print("\nProcessing unknown faces...")

for filename in os.listdir(UNKNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {filename}...")
        image_path = os.path.join(UNKNOWN_FACES_DIR, filename)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding, TOLERANCE)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            top, right, bottom, left = face_location
            cv2.rectangle(image, (left, top), (right, bottom), BOX_COLOR, FONT_THICKNESS)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), BOX_COLOR, cv2.FILLED)
            cv2.putText(image, name, (left + 6, bottom - 6), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
            
            print(f"- Found {name} at location {face_location}")

        cv2.imshow(f'Result - {filename}', image)
        print(f"Displaying results for {filename}. Press any key to close this window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("\nFinished processing all images.")