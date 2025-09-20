import os
import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance as sp_distance
import cv2

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
ENCODINGS_FILE = "encodings_arcface.pkl"
MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 0.68 

# --- Font and Color Configuration for OpenCV ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)

# --- 1. Encode Known Faces (if encodings file doesn't exist) ---
if not os.path.exists(ENCODINGS_FILE):
    # This part remains the same
    print("No existing encodings file. Encoding known faces...")
    known_faces_encodings = []
    known_faces_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0]
            try:
                result = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME, enforce_detection=False)
                if result and len(result) > 0:
                    embedding = result[0]["embedding"]
                    known_faces_encodings.append(embedding)
                    known_faces_names.append(name)
                    print(f"Encoded {name}")
                else:
                    print(f"Warning: No face detected in {filename}. Skipping.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_faces_encodings, "names": known_faces_names}, f)
    print(f"Encodings saved to {ENCODINGS_FILE}")

# --- 2. Load Encodings and Recognize Unknown Faces ---
print("\nLoading saved ArcFace encodings...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)
known_faces_encodings = data["encodings"]
known_faces_names = data["names"]
print(f"Loaded {len(known_faces_names)} known faces.")

print("\nRecognizing and visualizing faces in unknown images...")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(UNKNOWN_FACES_DIR, filename)
        print(f"\n--- Processing {filename} ---")

        # Load the image with OpenCV to draw on it
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        try:
            unknown_results = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME, enforce_detection=False)

            if not unknown_results:
                print("No faces detected.")
            
            for unknown_result in unknown_results:
                unknown_embedding = unknown_result["embedding"]
                face_location = unknown_result["facial_area"]
                
                min_distance = float('inf')
                best_match_name = "Unknown"

                for i, known_embedding in enumerate(known_faces_encodings):
                    distance = sp_distance.cosine(unknown_embedding, known_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        best_match_name = known_faces_names[i]
                
                # Determine the final name and box color
                if min_distance <= DISTANCE_THRESHOLD:
                    final_name = best_match_name
                    box_color = (0, 255, 0) # Green for a match
                    print(f"Match found: {final_name} (Distance: {min_distance:.4f})")
                else:
                    final_name = "Unknown"
                    box_color = (0, 0, 255) # Red for unknown
                    print(f"No close match found. (Closest: {best_match_name}, Dist: {min_distance:.4f})")

                # --- Draw on the image ---
                x, y, w, h = face_location['x'], face_location['y'], face_location['w'], face_location['h']
                
                # Draw the main bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), box_color, FONT_THICKNESS)
                
                # Draw a filled rectangle for the text label background
                cv2.rectangle(image, (x, y + h - 35), (x + w, y + h), box_color, cv2.FILLED)
                
                # Put the name text on the label
                cv2.putText(image, final_name, (x + 6, y + h - 6), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
            
            # --- Display the final image ---
            cv2.imshow(f"Result - {filename}", image)

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

print("\nFinished processing. Press any key in an image window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()