# multicore_recognition.py

import face_recognition
import pickle
import cv2
import multiprocessing as mp
import time

def process_frame_worker(input_queue, output_queue, known_face_data):
    """
    Worker function to process frames from the input queue.
    """
    # Unpack the known face data
    known_encodings = known_face_data["encodings"]
    known_names = known_face_data["names"]
    
    while True:
        # Get a frame from the input queue
        frame = input_queue.get()

        # Check for the sentinel value to exit
        if frame is None:
            break

        # --- HEAVY LIFTING IS DONE HERE ---
        # Resize frame for faster processing (optional, but recommended)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame
        boxes = face_recognition.face_locations(rgb_small_frame, model="hog")
        encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.55)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    name = known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            
            names.append(name)
        
        # Put the original frame and the results into the output queue
        output_queue.put((frame, boxes, names))

if __name__ == "__main__":
    # This check is essential for multiprocessing to work correctly
    
    print("[INFO] Loading encodings...")
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)

    # --- SETUP MULTIPROCESSING ---
    # Create queues for communication between processes
    input_q = mp.Queue(maxsize=5)
    output_q = mp.Queue(maxsize=5)

    # Define number of worker processes
    # A good rule of thumb is (number of cores - 1)
    NUM_WORKERS = 5 
    pool = []

    print(f"[INFO] Starting {NUM_WORKERS} worker processes...")
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=process_frame_worker,
            args=(input_q, output_q, data)
        )
        p.daemon = True
        p.start()
        pool.append(p)

    # --- MAIN PROCESS ---
    print("[INFO] Starting video stream...")
    video_capture = cv2.VideoCapture(0)
    
    # For FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Put the frame into the input queue for the workers to process
        # Use a non-blocking put to avoid waiting if the queue is full
        if not input_q.full():
            input_q.put(frame)

        # Get the processed results from the output queue
        # Use a non-blocking get to avoid waiting if the queue is empty
        try:
            processed_frame, boxes, names = output_q.get_nowait()
            
            # Draw the results on the processed frame
            for (top, right, bottom, left), name in zip(boxes, names):
                # Scale back up face locations since the frame they were detected on was resized
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(processed_frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # Update FPS counter
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1:
                fps = fps_frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                fps_frame_count = 0
                fps_start_time = time.time()

            cv2.imshow("Office Floor Monitoring", processed_frame)

        except Exception:
            # Queue was empty, just continue
            pass

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- CLEAN UP ---
    print("[INFO] Cleaning up processes...")
    # Send a signal to all workers to exit
    for _ in pool:
        input_q.put(None)
    
    # Wait for all workers to finish
    for p in pool:
        p.join()

    video_capture.release()
    cv2.destroyAllWindows()