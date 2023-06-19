#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 21:24:49 2023

@author: dwi
"""

import mtcnn
import os
import pandas as pd
import matplotlib.pyplot as plt
import face_recognition
from skimage import exposure
import numpy as np
from PIL import Image
import api
from numpy import asarray
import math
from itertools import combinations
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2

def repairRotation(img):
    print("recostruct image...(rotasi 90")
    img  = Image.fromarray(img)
    img  = np.array(img.rotate(90))
    plt.imshow(img)
    plt.show()
    return img

def repairWithClahe(img):
    print("recostruct image...hist")
    img  = exposure.equalize_adapthist(img/255)
    img  = img*255
    img  = img.astype(np.uint8)  
    plt.imshow(img)
    plt.show()
    return img

def rekonstruksiImage(img):
    img2=repairWithClahe(img)
    face_bounding_boxes = face_recognition.face_locations(img2)   
    if len(face_bounding_boxes)==0:
        img2=repairRotation(img)
        face_bounding_boxes = face_recognition.face_locations(img2)
        if len(face_bounding_boxes)==0:
            img2=img
    return img2,face_bounding_boxes

def getFace(img,face_location):
    top, right, bottom, left = face_location
    # plt.imshow(img[top:bottom, left:right])
    # plt.show()                                                  
    detected_face  = img[top:bottom, left:right] 
    return detected_face


def resizeImg(img,required_size=(224, 224)):
    face_image = Image.fromarray(img)
    face_image = face_image.resize(required_size)
    face_array = asarray(face_image)
    
    return face_array

DIM = 128
RGB = 3

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def alignment_procedure(img, left_eye, right_eye):
    #this function aligns given face in img based on left and right eye coordinates
     
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
     
    #-----------------------
    #find rotation direction
     
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
     
    #-----------------------
    #find length of triangle edges
     
    a = euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = euclidean_distance(np.array(right_eye), np.array(left_eye))
     
    #-----------------------
     
    #apply cosine rule
     
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
     
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree
     
        #-----------------------
        #rotate base image
     
        if direction == -1:
            angle = 90 - angle
     
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
     
    #-----------------------
     
    return img #return img anyway

def alignFaceMTCNN(img):
    align_faces=[]    
    detections = detector.detect_faces(img)
    #check dua wajah, nanti pakai face sj
    for detection in detections:
        score = detection["confidence"]        
        if score > 0.9:            
            x, y, w, h = detection["box"]
            detected_face = img[abs(y):abs(y+h), abs(x):abs(x+w)]
            keypoints = detection["keypoints"]
            left_eye = keypoints["left_eye"]
            right_eye = keypoints["right_eye"]            
            align_face = alignment_procedure(detected_face, left_eye, right_eye)                                                                         
            if len(align_face)!=0:
                align_face = resizeImg(align_face) 
                align_faces.append(align_face)
                print(score)
                # plt.imshow(detected_face)
                # plt.show()                      
        # break
    return align_faces

def pushToArrImg(align_faces,file,count):    
    no_align_faces=True
    for align_face in align_faces:
        
        no_align_faces=False
        face_array = resizeImg(align_face) 
        face_images.append(face_array)                                              
        file_images.append(file)
        id_numbers.append(count)
        count=count+1
        
    return face_images,no_align_faces,count

def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')

    # perform prediction
    return model(samples)


if __name__ == '__main__':
    path = "lfw\Abdullah_Gul"
    detector = mtcnn.MTCNN()
    idfile = 0
    needAlign = True

    for dirpath, subdirs, files in os.walk(path):
        # print(files)
        face_images,file_images,id_numbers,count = [],[],[],0
        for file in files:
            if ".jpg" in file:                
                idfile = idfile + 1
                # count=count+1
                noFace = True
                print(file)
                image_path = os.path.join(dirpath, file)
                img = plt.imread(image_path)
                face_bounding_boxes = face_recognition.face_locations(img)

                # deteksi menggunakan face_recognition
                if len(face_bounding_boxes) == 0:
                    print("wajah tidak terdeteksi")
                    img, face_bounding_boxes = rekonstruksiImage(img)
                for face_location in face_bounding_boxes:
                    detected_face = getFace(img, face_location)
                    plt.imshow(detected_face)
                    plt.show()
                    if needAlign:
                        align_faces = api.face_alignment(detected_face)
                        face_images, no_align_faces,count = pushToArrImg(
                            align_faces,file,count)
                        if no_align_faces == False:
                            noFace = False
                    else:
                        face_array = resizeImg(detected_face)
                        face_images.append(face_array)
                        noFace = False                    
                # face recognition gagal, digunakan MTCNN
                if noFace == True:
                    print("face tdk dapat di align, try with mtcnn")
                    align_faces = alignFaceMTCNN(img)
                    # face_images.extend(align_faces)
                    face_images, no_align_faces,count = pushToArrImg(
                        align_faces,file,count)
                print("Add face face_images->%d"%len(face_images))
                ## array ke 16 dan 17 bukan abdulah gul mestinya di exclude
            
        scoreSimilar = get_model_scores(face_images)   
        comb = combinations(id_numbers, 2)
        similars = np.zeros(len(id_numbers))        
        for c in comb:
            diff = cosine(scoreSimilar[c[0]], scoreSimilar[c[1]])
            if diff <= 0.4:
                print("sama match antara idx=%d vs idx=%d"%(c[0],c[1]))
                similars[c[0]] += 1
                similars[c[1]] += 1   
            else:
                print("tidak match antara idx=%d vs idx=%d"%(c[0],c[1]))
                
            
        df = pd.DataFrame({"id": np.array(id_numbers), "mirip": similars, "file":np.array(file_images)})
        df = df.sort_values(by=["mirip"], ascending=False)        
        df.reset_index(drop=True, inplace=True)
        print("Hapus yang miripnya dibawah 1 ")
        tidak_mirip = df[df['mirip']<1]
        print(tidak_mirip)
        print("yang dipakai train adalah ")
        mirip = df[df["mirip"] > 1] 
        print(mirip)
        print("data wajah yang tertangkap sejumlah =%d dari jumlah file=%d"%(len(df),idfile))
        for i in range(len(tidak_mirip)):
            plt.imshow(face_images[tidak_mirip['id'].values[i]])
            plt.show()
        for i in range(len(tidak_mirip)):
            del face_images[tidak_mirip['id'].values[i]]
        print("setelah dihapus menjadi =%d face dari sejumlah files=%d"%(len(mirip),len(mirip['file'].unique())))
        
        
            
        
        
        
        



            
                
                