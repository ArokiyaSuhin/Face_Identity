import face_recognition
import cv2
import os
import numpy as np

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6  # Lower is more strict. 0.6 is a good default.
MODEL = "cnn"  # or "cnn" for more accurate but slower processing
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0) # Green
TEXT_COLOR = (255, 255, 255) # White

print("Loading known faces...")

# --- Load Known Faces ---
known_faces_encodings = []
known_faces_names = []

# Iterate through each file in the known_faces directory
for filename in os.listdir(KNOWN_FACES_DIR):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Load the image
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings (assuming one face per image for known faces)
            # The [0] is to get the first face found in the image.
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_faces_encodings.append(encodings[0])
                # Use the filename (without extension) as the person's name
                known_faces_names.append(os.path.splitext(filename)[0])
            else:
                print(f"Warning: No face found in {filename}. Skipping.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if not known_faces_encodings:
    print("No known faces were loaded. Please check the 'known_faces' directory.")
    exit()

print(f"Loaded {len(known_faces_names)} known faces.")

# --- Process Unknown Faces ---
print("\nProcessing unknown faces...")

# Iterate through each file in the unknown_faces directory
for filename in os.listdir(UNKNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {filename}...")
        
        # Load the unknown image
        image_path = os.path.join(UNKNOWN_FACES_DIR, filename)
        image = cv2.imread(image_path)
        
        # Find all face locations and encodings in the current image
        # Using cv2 to read the image allows for easy drawing later
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Loop through each face found in the unknown image
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the unknown face with all known faces
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding, TOLERANCE)
            name = "Unknown" # Default name if no match is found

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            # Draw a box around the face and label it
            top, right, bottom, left = face_location
            
            # Draw the bounding box
            cv2.rectangle(image, (left, top), (right, bottom), BOX_COLOR, FONT_THICKNESS)
            
            # Create a filled rectangle for the name label
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), BOX_COLOR, cv2.FILLED)
            
            # Put the name text on the label
            cv2.putText(image, name, (left + 6, bottom - 6), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
            
            print(f"- Found {name} at location {face_location}")

        # Display the resulting image
        cv2.imshow(f'Result - {filename}', image)
        print(f"Displaying results for {filename}. Press any key to close this window and continue.")
        cv2.waitKey(0) # Wait for a key press to show the next image
        cv2.destroyAllWindows()

print("\nFinished processing all images.")