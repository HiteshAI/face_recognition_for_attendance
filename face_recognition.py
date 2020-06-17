import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image
import argparse
import pickle
import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def get_faces_cropped_from_input(detector, input_dir):
    required_size = (160,160)
    img_file_lst = os.listdir(input_dir)
    cropped_faces_array_list = []

    for img_file in img_file_lst:
        path = input_dir + "/" + img_file
        img = plt.imread(path)
        # detect faces in the image
        face = detector.detect_faces(img)
        if face:
            bbox = face[0]['box']
            x1, y1, width, height = bbox
            x2, y2 = x1 + width, y1 + height
            cropped_face = img[y1:y2, x1:x2]
            plt.imshow(cropped_face)
            cropped_face_array = Image.fromarray(cropped_face)
            cropped_face_array = cropped_face_array.resize(required_size)
            cropped_face_array = asarray(cropped_face_array)
            cropped_faces_array_list.append(cropped_face_array)

    return cropped_faces_array_list

def get_faces_cropped_from_videos_prediction(detector, img):
    required_size = (160, 160)
    # img_file_lst = os.listdir(predict_dir)
    cropped_faces_array_list = []
    face_location = []
    # img = plt.imread(path)


    # detect faces in the image
    face = detector.detect_faces(img)
    for face in face:
        if face['confidence'] > 0.90:
            bbox = face['box']
            x1, y1, width, height = bbox
            x2, y2 = x1 + width, y1 + height
            cropped_face = img[y1:y2, x1:x2]
            plt.imshow(cropped_face)
            cropped_face_array = Image.fromarray(cropped_face)
            cropped_face_array = cropped_face_array.resize(required_size)
            cropped_face_array = asarray(cropped_face_array)
            cropped_faces_array_list.append(cropped_face_array)
            face_location.append(face)

    dict = {
        "face_location": face_location,
        "cropped_faces_array_list": cropped_faces_array_list
    }

    return cropped_faces_array_list, face_location, dict

def get_faces_cropped_from_prediction(detector, path):
    required_size = (160, 160)
    # img_file_lst = os.listdir(predict_dir)
    cropped_faces_array_list = []
    face_location = []
    img = plt.imread(path)


    # detect faces in the image
    face = detector.detect_faces(img)
    for face in face:
        if face['confidence'] > 0.90:
            bbox = face['box']
            x1, y1, width, height = bbox
            x2, y2 = x1 + width, y1 + height
            cropped_face = img[y1:y2, x1:x2]
            plt.imshow(cropped_face)
            cropped_face_array = Image.fromarray(cropped_face)
            cropped_face_array = cropped_face_array.resize(required_size)
            cropped_face_array = asarray(cropped_face_array)
            cropped_faces_array_list.append(cropped_face_array)
            face_location.append(face)

    dict = {
        "face_location": face_location,
        "cropped_faces_array_list": cropped_faces_array_list
    }

    return cropped_faces_array_list, face_location, dict

def get_embedding(model, cropped_face_list):

    embeddibgs = []
    for cropped_face in cropped_face_list:
        cropped_face = cropped_face.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = cropped_face.mean(), cropped_face.std()
        cropped_face = (cropped_face - mean) / std
        # transform face into one sample
        samples = expand_dims(cropped_face, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        embeddibgs.append(yhat[0])
    return embeddibgs

def euclidean(x,y):
    return np.sqrt(np.sum((x-y)**2))

def get_face_names(input_dir):

    img_file_lst = os.listdir(input_dir)
    known_face_names = []
    for img_file in img_file_lst:
        img_name = img_file.rsplit(".", 1)[0]
        known_face_names.append(str(img_name))
    return known_face_names

def save_input_feature_embedding(inputs_embedding, known_faces):

    x = zip(known_faces, inputs_embedding)
    features = []
    for itr in x:
        features.append({itr[0]:itr[1]})


    with open('input_data.p', 'wb') as fp:
        pickle.dump(features, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_input_feature_embedding():
    with open('input_data.p', 'rb') as fp:
        data = pickle.load(fp)
    return data



def face_matching_from_image(img_path, predict_embedding, data, face_array_and_loc, tolerance):
    img = plt.imread(img_path)
    img_name = os.path.basename(img_path)
    for idx1, emb in enumerate(predict_embedding):
            name = 'Unknown'
            matches = []
            for d in data:
                for k,v in d.items():
                    dist = euclidean(emb, v)
                    matches.append({k:dist})

                # index_min = np.argmin(matches)
                for m in matches:
                    for k,v in m.items():
                        if v <= int(tolerance):
                            name = str(k)
            x, y, width, height  = face_array_and_loc['face_location'][idx1]['box']
            image = cv2.rectangle(img, (x, y), (x + width, y + height), (255, 36, 12), 1)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 36, 12), 1)

    plt.imshow(image)
    plt.show()
    cv2.imwrite('./output_dir/'+ img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def face_matching_from_video(img_path, predict_embedding, data, face_array_and_loc, tolerance):
    img = plt.imread(img_path)
    img_name = os.path.basename(img_path)
    for idx1, emb in enumerate(predict_embedding):
            name = 'Unknown'
            matches = []
            for d in data:
                for k,v in d.items():
                    dist = euclidean(emb, v)
                    matches.append({k:dist})

                # index_min = np.argmin(matches)
                for m in matches:
                    for k,v in m.items():
                        if v <= int(tolerance):
                            name = str(k)
            x, y, width, height  = face_array_and_loc['face_location'][idx1]['box']
            image = cv2.rectangle(img, (x, y), (x + width, y + height), (255, 36, 12), 1)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 36, 12), 1)

    # plt.imshow(image)
    # plt.show()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('output_now.avi', fourcc, 29.97, (640, 360))
    output_movie.write(img)
    # cv2.imwrite('./output_dir_now_video/'+ img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def get_video_frame_recognition():

    input_movie = cv2.VideoCapture("videoplayback.mp4")
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an output movie file (make sure resolution/frame rate matches input video!)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))
    frame_number = 0
    while frame_number < 2500:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        img = frame[:, :, ::-1]
        # plt.imshow(img)
        cv2.imwrite("./output_dir_video/"+str(frame_number)+ ".jpeg", img)




if __name__ == '__main__':

    
    model = load_model('model/facenet_keras.h5')
    detector = MTCNN()
    data = load_input_feature_embedding()
    parser = argparse.ArgumentParser()
    parser.add_argument("select",
                    help="select training or inference", type=str)

    parser.add_argument("-img","--image",
                        help="image file name from predict dir", type=str)
    args = parser.parse_args()
    select = args.select

    if select == 'train':
        input_dir = './input_dir'
        input_faces_array_list = get_faces_cropped_from_input(detector, input_dir)
        known_faces = get_face_names(input_dir)
        inputs_embedding = get_embedding(model, input_faces_array_list)
        save_input_feature_embedding(inputs_embedding, known_faces)

    elif select == 'inference':
        predict_dir = './predict_dir'
        img_path = predict_dir + '/' + args.image
        print(args.image)
        predict_face_array_list, face_location, face_array_and_loc = get_faces_cropped_from_prediction(detector, img_path)
        data = load_input_feature_embedding()
        predict_embedding = get_embedding(model, predict_face_array_list)
        face_matching_from_image(img_path, predict_embedding, data, face_array_and_loc, tolerance=10)
        print("Image saved")

    elif select == 'video':
        # Get frames from video and save to directory
        # get_video_frame_recognition()
        predict_dir = './output_dir_video'
        # lsorted = sorted(l, key=lambda x: int(os.path.splitext(x)[0]))
        for img_file in (os.listdir(predict_dir)):
            img_path = predict_dir + '/' + img_file
            print(img_path)
            predict_face_array_list, face_location, face_array_and_loc = get_faces_cropped_from_prediction(detector, img_path)
            data = load_input_feature_embedding()
            predict_embedding = get_embedding(model, predict_face_array_list)
            face_matching_from_video(img_path, predict_embedding, data, face_array_and_loc, tolerance=10)
            print("Video saved")
