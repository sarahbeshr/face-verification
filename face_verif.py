import face_recognition
import numpy as np
import cv2
import math
import os


def face_distance_to_conf(face_distance, face_match_threshold=0.8):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

dirname = os.path.dirname(__file__)

def crop_image(image):
    faceCascade = cv2.CascadeClassifier(os.path.join(dirname, "haarcascade_frontalface_default"))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(
            rgb,
            scaleFactor=1.3,
            minNeighbors=2,
            minSize=(270, 270))
    if len(faces) == 0:
        return rgb
    else:
        for (x, y, w, h) in faces:
            roi = rgb[y:y + h, x:x + w]
        return roi

    #Image input as numpy array

image1 = face_recognition.load_image_file(os.path.join(dirname, "Data/Jon/jon1"))
image2 = face_recognition.load_image_file(os.path.join(dirname, "Data/Jon/jon2"))

Image1_ROI = crop_image(image1)
Image2_ROI = crop_image(image2)

#128d encoding
encoding1 = face_recognition.api.face_encodings(Image1_ROI, known_face_locations=None, num_jitters=5, model='large')
encoding2 = face_recognition.api.face_encodings(Image2_ROI, known_face_locations=None, num_jitters=5, model='large')
if len(encoding1) == 0 or len(encoding2)==0:
    print("Encoding error")
else:
    ID_encoding = encoding1[0]
    Selfie_encoding = encoding2[0]
    face_distance = face_recognition.api.face_distance([ID_encoding], Selfie_encoding)
    face_match_percentage = face_distance_to_conf(face_distance)
    face_match_percentage = (np.round(face_match_percentage, 2))[0]
    face_match_percentage = "{:.0%}".format(face_match_percentage)
    if face_distance < 0.6:
        result  = "Match"
    else:
        result = "Not a match"

Dict = {'Decision': result,
        'Match Percentage': face_match_percentage}
print(Dict)

