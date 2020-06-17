# install dlib (will take 10 or more minutes)
# !apt update
# !apt install -y cmake
# !pip3 install dlib
# !sudo pip install -v --install-option="--no" --install-option="DLIB_USE_CUDA" dlib
# !pip3 install face_recognition

#
# from PIL import Image, ImageDraw
# from IPython.display import display

import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import os

def display_faces_from_dir(face_dir):
    faces = os.listdir(face_dir)
    for face in faces:
        path = str(face_dir) + str(face)
        pil_face = Image.open(path)
        display(pil_face)


        # plt.show()

def get_faces_from_dir(face_dir):
    faces = os.listdir(face_dir)
    return faces

def get_face_names(faces):
    # TODO strip after.

    known_face_names = []
    for face in faces:
        # import re
        # face =  re.sub('^.*?. ', '', face)
        face = face.rsplit(".", 1)[0]
        known_face_names.append(str(face))
    return known_face_names

def face_encoding(face):
    path = str(face_dir) + str(face)
    load_image = face_recognition.load_image_file(path)
    face_encoding = face_recognition.face_encodings(load_image)[0]
    return face_encoding



def face_recognition_image(detect_imag, known_face_names,known_face_encodings):
    unknown_image = face_recognition.load_image_file(detect_imag)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)


    pil_image = Image.fromarray(unknown_image)

    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]


        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top-5), (right, bottom)), outline=(255, 0, 0))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name, spacing=4)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=1)


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    import matplotlib.plt as plt
    plt.imshow(pil_image)
    plt.show()
    # display(pil_image)

def face_recognition_video(known_face_names, known_face_encodings):
    input_movie = cv2.VideoCapture("videoplayback.mp4")
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))
    frame_number = 0

    while frame_number < 1500:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"








            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)



if __name__ == '__main__':
    # print(os.getcwd())
    # os.chdir(os.getcwd())
    detect_imag = 'detect.jpg'

    known_face_encodings = []
    # known_face_names = []
    face_dir = './faces/'

    # display_faces_from_dir(face_dir)
    faces = get_faces_from_dir(face_dir)
    known_face_names = get_face_names(faces)
    for face in faces:
        encoding = face_encoding(face)
        known_face_encodings.append(encoding)
    print('Learned encoding for', len(known_face_encodings), 'images.')

    # face_recognition_image(detect_imag, known_face_names, known_face_encodings)

    # face_recognition_video(known_face_names, known_face_encodings)



