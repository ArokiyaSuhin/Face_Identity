import face_recognition
import numpy as np

# --- NO LONGER NEEDED ---
# from PIL import Image
# def load_image_as_rgb(path):
#    ...
# ------------------------

try:
    # Use the library's built-in loader, which handles all conversions automatically
    img1 = face_recognition.load_image_file(r"D:\Face_Identity\Reserved_ImageAttachment_[6]_[Images][32]_[92a96715fc1748a1a0201be11dd9f039][1]_[1].png")
    img2 = face_recognition.load_image_file(r"D:\Face_Identity\Reserved_ImageAttachment_[6]_[Images][32]_[d5252f8dad3447508ed3e163ae62a523][1]_[1].png")
    print("✅ Images loaded successfully.")
    
    # Get encodings
    # This part should work now that the images are loaded correctly
    enc1_list = face_recognition.face_encodings(img1)
    enc2_list = face_recognition.face_encodings(img2)
    
    if not enc1_list:
        print("❌ No face found in first image")
    elif not enc2_list:
        print("❌ No face found in second image")
    else:
        # Get the first face encoding from each image
        enc1 = enc1_list[0]
        enc2 = enc2_list[0]
    
        # Compare faces and get the distance
        results = face_recognition.compare_faces([enc1], enc2)
        distance = face_recognition.face_distance([enc1], enc2)
    
        # Display results
        if results[0]:
            print(f"✅ Same person (Similarity Score: {1 - distance[0]:.2f})")
        else:
            print(f"❌ Different persons (Similarity Score: {1 - distance[0]:.2f})")

except FileNotFoundError as e:
    print(f"❌ ERROR: File not found. Please check the path.")
    print(e)


