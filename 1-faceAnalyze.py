import torch
import cv2
import face_recognition
from PIL import Image, ImageDraw

# load jpg file into a numpy array
image = face_recognition.load_image_file("E:/project/project cv/1.人脸打卡持续学习/Data/hard/5-y.jpg")

#在图片中定位人脸位置
# face_location = face_recognition.face_locations(image)
# face_locations is now an array listing the co-ordinates of each face!

# 识别人脸关键点，包括眼睛、鼻子、嘴和下巴
# find all facialfeatures in all the faces in the image
face_landarks_list = face_recognition.face_landmarks(image)

print("I found {} faes in the photograph.".format(len(face_landarks_list)))

# create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landarks_list:
    # print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # let us trace out each facial feature in the image with a line
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

# show the picture
pil_image.show()

# 身份识别
# known_image = face_recognition.load_image_file("")
# unknown_image = face_recognition.load_image_file("")
#
# known_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#
# results = face_recognition.compare_faces([known_encoding], unknown_encoding)