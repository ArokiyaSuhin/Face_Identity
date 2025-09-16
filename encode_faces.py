# encode_faces.py
import face_recognition
import os
import pickle

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"
MODEL = "hog" # or "cnn"

print("Starting to process known faces...")

# --- Load and Encode Known Faces ---
known_faces_encodings = []
known_faces_names = []

# Iterate through each person in the known_faces directory
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Load image
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Find face encodings. We assume one face per image.
            # The [0] gets the encoding for the first face found.
            encodings = face_recognition.face_encodings(image, model=MODEL)
            
            if len(encodings) > 0:
                known_faces_encodings.append(encodings[0])
                # Get the name from the filename (without extension)
                name = os.path.splitext(filename)[0]
                known_faces_names.append(name)
                print(f"Processed {name}")
            else:
                print(f"Warning: No face found in {filename}. Skipping.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save the encodings and names to a file
print("\nSaving encodings to disk...")
data = {"encodings": known_faces_encodings, "names": known_faces_names}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)
    
print(f"Encodings saved successfully to '{ENCODINGS_FILE}'")