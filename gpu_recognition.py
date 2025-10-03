import cv2
import pickle
import time
import face_recognition
from threading import Thread

# (The WebcamStream class is the same as before)
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True
        self.stream.release()

# --- SCRIPT START ---
print("[INFO] Loading encodings...")
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

print("[INFO] Starting threaded video stream...")
vs = WebcamStream(src=0).start()

# --- BATCH PROCESSING SETUP ---
BATCH_SIZE = 4  # Process frames in batches of 4. Tune this number.
frame_batch = []
rgb_small_frame_batch = []
batch_start_time = time.time()
fps = 0

# --- MAIN PROCESSING LOOP ---
while True:
    frame = vs.read()
    if frame is None:
        continue

    # --- CPU PRE-PROCESSING ---
    # Resize and convert frame, then add to the batch
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    frame_batch.append(frame)
    rgb_small_frame_batch.append(rgb_small_frame)

    # When the batch is full, process it
    if len(frame_batch) == BATCH_SIZE:
        # --- GPU BATCH PROCESSING ---
        # This is where the GPU does its work on all frames at once
        batch_of_boxes = face_recognition.batch_face_locations(rgb_small_frame_batch, number_of_times_to_upsample=0, batch_size=BATCH_SIZE)

        # --- POST-PROCESSING AND DISPLAY ---
        # We'll display the LAST frame of the batch with the results
        last_frame_in_batch = frame_batch[-1]
        
        # Loop through the results for each frame in the batch
        for frame_index_in_batch, boxes in enumerate(batch_of_boxes):
            if not boxes: continue # Skip if no faces were found in this frame
            
            # For simplicity, we'll just draw boxes found in the last frame
            if frame_index_in_batch == BATCH_SIZE - 1:
                encodings = face_recognition.face_encodings(rgb_small_frame_batch[frame_index_in_batch], boxes)
                names = []
                for encoding in encodings:
                    matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.55)
                    name = "Unknown"
                    if True in matches:
                        # ... (name matching logic is the same) ...
                        matched_idxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for i in matched_idxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)
                    names.append(name)
                
                # Draw results on the last frame
                for (top, right, bottom, left), name in zip(boxes, names):
                    top *= 2; right *= 2; bottom *= 2; left *= 2
                    cv2.rectangle(last_frame_in_batch, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(last_frame_in_batch, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Calculate and display FPS based on batch processing time
        elapsed_time = time.time() - batch_start_time
        fps = BATCH_SIZE / elapsed_time
        cv2.putText(last_frame_in_batch, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Office Floor Monitoring (GPU Batch)", last_frame_in_batch)

        # Reset for the next batch
        frame_batch = []
        rgb_small_frame_batch = []
        batch_start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEAN UP ---
print("[INFO] Cleaning up...")
vs.stop()
cv2.destroyAllWindows()