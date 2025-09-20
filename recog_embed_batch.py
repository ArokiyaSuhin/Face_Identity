import face_recognition
import cv2
import os
import numpy as np
import multiprocessing
import time

# --- Configuration (remains the same) ---
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6
MODEL = "hog"  # Using "hog" as it's CPU-focused. CNN is extremely slow on CPU.
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

# --- Worker Function ---
# This function will be run in parallel on each CPU core.
# It processes a single unknown image.
def process_single_image(image_path_tuple):
    # Unpack tuple to get path and known face data
    image_path, known_faces_encodings, known_faces_names = image_path_tuple
    filename = os.path.basename(image_path)
    print(f"Processing {filename} on process {os.getpid()}...")

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

    # Return the processed image and its original filename for display
    return filename, image

# --- Main Execution Block ---
if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly.

    # 1. Load Known Faces (this happens once in the main process)
    print("Loading known faces...")
    known_faces_encodings = []
    known_faces_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            # Find locations first, especially important for CNN model
            locations = face_recognition.face_locations(image, model=MODEL)
            if locations:
                encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
                known_faces_encodings.append(encoding)
                known_faces_names.append(os.path.splitext(filename)[0])

    print(f"Successfully encoded {len(known_faces_names)} known faces.")

    # 2. Prepare the list of tasks for the worker processes
    unknown_image_paths = [os.path.join(UNKNOWN_FACES_DIR, f) for f in os.listdir(UNKNOWN_FACES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Bundle the image path with the known face data that each process will need
    tasks = [(path, known_faces_encodings, known_faces_names) for path in unknown_image_paths]

    # 3. Create a pool of processes and distribute the work
    print(f"\nStarting parallel processing on {os.cpu_count()} CPU cores...")
    start_time = time.time()
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        # pool.map distributes the 'tasks' list to the 'process_single_image' function
        results = pool.map(process_single_image, tasks)

    end_time = time.time()
    print(f"Finished processing all images in {end_time - start_time:.2f} seconds.")

    # 4. Display the results collected from all processes
    print("\nDisplaying all results.")
    for filename, processed_image in results:
        cv2.imshow(f'Result - {filename}', processed_image)

    print("Press any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()