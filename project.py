import face_recognition
import cv2
import os
import numpy as np


dataset_path = "face-recognition\Face-Recognition-Project\images"  # Replace this with your actual path

# Function to load images and extract facial encodings
def load_images_and_encodings(folder):
    encodings = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                name = os.path.splitext(filename)[0]
                names.append(name)
    return encodings, names

# Load known faces and encodings
known_face_encodings, known_face_names = load_images_and_encodings(dataset_path)

# Initialize webcam
video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over each face found in the frame to see if it's someone we know
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name above the face
        label = f"{name}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 1)
        label_x = left
        label_y = top - 10  

        # Ensure text is visible if it goes above the top edge of the image
        if label_y < 20:
            label_y = top + 20

        cv2.rectangle(frame, (label_x, label_y - label_size[1] - 10), (label_x + label_size[0], label_y), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' on the keyboard to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
