#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 04:16:02 2021

@author: dwi
"""
import os,gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import face_recognition
from skimage import exposure
from PIL import Image
import api
from numpy import asarray
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from itertools import combinations
from scipy.spatial.distance import cosine
import mtcnn,math
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation, Convolution2D,Dense,
    Dropout, BatchNormalization, MaxPooling2D,     
    Flatten
    )

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import pickle

DIM=128#64#->48
RGB=3
input_shape = (DIM, DIM, RGB)
size = (DIM, DIM)

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


def repairWithClahe(img):
    print("recostruct image...hist")
    img  = exposure.equalize_adapthist(img/255)
    img  = img*255
    img  = img.astype(np.uint8)  
    plt.imshow(img)
    plt.show()
    return img

def repairRotation(img):
    print("recostruct image...(rotasi 90")
    img  = Image.fromarray(img)
    img  = np.array(img.rotate(90))
    plt.imshow(img)
    plt.show()
    return img

def getFace(img,face_location):
    top, right, bottom, left = face_location
    # plt.imshow(img[top:bottom, left:right])
    # plt.show()                                                  
    detected_face  = img[top:bottom, left:right] 
    return detected_face

def rekonstruksiImage(img):
    img2=repairWithClahe(img)
    face_bounding_boxes = face_recognition.face_locations(img2)   
    if len(face_bounding_boxes)==0:
        img2=repairRotation(img)
        face_bounding_boxes = face_recognition.face_locations(img2)
        if len(face_bounding_boxes)==0:
            img2=img
    return img2,face_bounding_boxes

def resizeImg(img,required_size=(224, 224)):
    face_image = Image.fromarray(img)
    face_image = face_image.resize(required_size)
    face_array = asarray(face_image)
    
    return face_array

def getResize(images):
    size   = (DIM, DIM)
    ftrImg = []
    for i in range(len(images)):
        img = images[i]   
        img = resizeImg(img,size)
        # img = resizeImg(img,size)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        
        # img = cv2.resize(img, size)
        # img = img.reshape(DIM,DIM,RGB)
        ftrImg.append(img)
    ftrImg = np.array(ftrImg)      
    return ftrImg

def pushToArrImg(align_faces,face_images):
    no_align_faces=True
    for align_face in align_faces:
        no_align_faces=False
        face_array = resizeImg(align_face) 
        face_images.append(face_array)                                              
    return face_images,no_align_faces

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

def extract_face_from_image(image_path, required_size=(224, 224),needAlign=True):
  # load image and detect faces
    noFace=True
    face_images = []
    img = plt.imread(image_path)
    face_bounding_boxes = face_recognition.face_locations(img)   
    if len(face_bounding_boxes)==0:
        print("wajah tidak terdeteksi")
        img,face_bounding_boxes = rekonstruksiImage(img)    
    for face_location in face_bounding_boxes:
        detected_face = getFace(img,face_location)
        # plt.imshow(detected_face)
        # plt.show()
        if needAlign:
            align_faces    = api.face_alignment(detected_face)            
            face_images,no_align_faces = pushToArrImg(align_faces,face_images)
            if no_align_faces==False:
                noFace=False                
        else:
            face_array = resizeImg(detected_face) 
            face_images.append(face_array)
            noFace=False
    if noFace==True :
        print("face tdk dapat di align, try with mtcnn")                                  
        align_faces = alignFaceMTCNN(img)
        face_images,no_align_faces = pushToArrImg(align_faces,face_images)
            

    return face_images

def getEncoding(detected_face):
    face_endcoding = face_recognition.face_encodings(detected_face)                    
    if len(face_endcoding)==0:
        print("wajah tidak dapat di encoding")        
        plt.imshow(detected_face)
        plt.show()      
    return face_endcoding  

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

def loadImageDetectFace(path, fr=False):    
    ftr,nmFiles,idImgs,scoreFace,idNumber  =[],[],[],[],0    
    for nmFile in os.listdir(path):
        print("nmFile ...." + nmFile)
        image_path = path+"/"+nmFile
        extracted_face = extract_face_from_image(image_path, 
                                                 required_size=(224, 224))                
        if len(extracted_face)>0:
            if fr==True:
                for detected_face in extracted_face:
                    fe = getEncoding(detected_face)
                    if len(fe)>0:
                        scoreFace.append(fe)
                        ftr.append(detected_face)                      
                        nmFiles.append(nmFile)      
                        idImgs.append(idNumber)
                        idNumber+=1  
            else:
                model_scores   = get_model_scores(extracted_face)        
                for i in range(len(model_scores)):
                    scoreFace.append(model_scores[i])
                    ftr.append(extracted_face[i])                      
                    nmFiles.append(nmFile)      
                    idImgs.append(idNumber)
                    idNumber+=1
                    
               
    comb      = combinations(idImgs, 2)
    similars  = np.zeros(len(idImgs))
    for c in comb:
        # print(c)
        matches=False
        if fr==True :
            matches = face_recognition.compare_faces(np.array(scoreFace[c[0]]), 
                                                     np.array(scoreFace[c[1]]))

        else:
            diff=cosine(scoreFace[c[0]], scoreFace[c[1]])
            print(diff)
            if  diff <= 0.4:
                matches=True
            
        if matches:
            print("Faces Matched")                            
            similars[c[0]]+=1
            similars[c[1]]+=1
        else:
            print("ada yg tidak match")
            plt.imshow(ftr[c[0]])
            plt.show()      
            plt.imshow(ftr[c[1]])
            plt.show()      

    df = pd.DataFrame({'id':np.array(idImgs), 'mirip':similars})          
    df = df.sort_values(by=['mirip'],ascending=False)  
    df = df[df['mirip']>1]
    df.reset_index(drop=True, inplace=True)
    #get base on 3 folder->3 face sj
    # selectedFtr,selectedLbl,selectedUsia,selectNm=[],[],[],[]
    selectedFtr,selectNm=[],[]
    cntFace=0
    for d in range(len(df)):        
        idImg = df['id'].loc[d]        
        if nmFiles[idImg] not in selectNm:
            selectedFtr.append(ftr[idImg])
            # selectedLbl.append(jk)               
            # selectedUsia.append(umur)
            selectNm.append(nmFiles[idImg])
            cntFace+=1
        if cntFace >= 3:
            break            
            
    # return selectedFtr,selectedLbl,selectedUsia,selectNm   
    return selectedFtr,selectNm   

def get_callbacks(name_weights, patience_lr):
    early_stopper = EarlyStopping(patience=20,monitor='val_loss',mode='auto')
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_accuracy', mode='auto')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.01, patience=patience_lr, verbose=1, min_delta=1e-4, mode='auto')
    return [mcp_save, reduce_lr_loss,early_stopper]


def simpler_CNN(input_shape, num_classes):
    clear_session()

    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(5, 5),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))

    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    
    return model

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Convolution2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Convolution2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Convolution2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


def alexNet(input_shape,num_class):
    clear_session()
    model = Sequential()
    
    model.add(Convolution2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    model.add(Convolution2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    model.add(Convolution2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Convolution2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(150, 150, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(num_class))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    model.summary()
    return model

def simpleCNN(input_shape,num_reg):
    clear_session()
    model = Sequential()
    model.add(Convolution2D(filters = 16, 
            kernel_size = (3, 3),                       
            input_shape = input_shape,             
            activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))    
    model.add(Convolution2D(32, 4, 4, activation = 'relu',
                            kernel_regularizer=regularizers.l2(l=0.01),
                            bias_regularizer=regularizers.l2(0.01)
                            ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))   
    model.add(Flatten())
    
    model.add(Dense(units = 1024, activation = 'relu',
            kernel_regularizer=regularizers.l2(0.001),
              ))
   
    model.add(Dense(units = 512, activation = 'relu',
            kernel_regularizer=regularizers.l2(0.001),
              ))
    
    model.add(Dense(units = 256, activation = 'relu',
            kernel_regularizer=regularizers.l2(0.001),
              ))


    # model.add(Dense(units = num_reg, activation = 'sigmoid'))    
    model.add(Dense(units = num_reg, activation = 'softmax'))    
    # sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True,decay=1e-6)
    # model.compile(optimizer = 'adam',
                    # loss = 'mse',
                    # loss=tf.keras.losses.Huber(delta=1.5),
                    # loss = tf.keras.losses.LogCosh(),
                                 # metrics = ['accuracy']
                  # metrics=['mae',r2,'mse']   
                                 # )
    
    model.summary()
    return model 


def detectFacesWithDetecttor(img):
    detections = detector.detect_faces(img)
    for detection in detections:
        score = detection["confidence"]        
        if score > 0.9:            
            return True
    return False
    
def getFaceAug(imgEH):   
    
    face_bounding_boxes = face_recognition.face_locations(imgEH)
    #detections = detector.detect_faces(imgEH)
    numFaceDetected = len(face_bounding_boxes)
    #print(numFaceDetected)
    try:
        if numFaceDetected >1:    
            print('Image face > 1 found') 
            
            return None   
        elif numFaceDetected ==1:
        
             for face_location in face_bounding_boxes:
                    top, right, bottom, left = face_location
                    detected_face  = imgEH[top:bottom, left:right] 
                    #print("ok")  
                    align_face     = api.face_alignment(detected_face)   
                    if len(align_face)!=0:
                        align_face = np.array(align_face)
                        align_face =align_face.reshape(align_face.shape[1],align_face.shape[2],align_face.shape[3])
                        if detectFacesWithDetecttor(align_face):
                            align_face = cv2.resize(align_face, size)
                            return align_face                        
                        else:
                            return None
                    else:
                         return None                   
        elif numFaceDetected==0:
            detections = detector.detect_faces(imgEH)
            #face_bounding_boxes = face_recognition.face_locations(imgEH)   
            if len(detections)==0:
               #print("face not found")
               imgRekon,face_bounding_boxes = rekonstruksiImage(imgEH)    
               imgEH = imgRekon
            detections = detector.detect_faces(imgEH)
            for detection in detections:
             score = detection["confidence"]        
             if score > 0.9:            
                 x, y, w, h = detection["box"]
                 detected_face = imgEH[abs(y):abs(y+h), abs(x):abs(x+w)]
                 keypoints = detection["keypoints"]
                 left_eye = keypoints["left_eye"]
                 right_eye = keypoints["right_eye"]            
                 align_face = alignment_procedure(detected_face, left_eye, right_eye)                                                                         
                 if detectFacesWithDetecttor(align_face):
                     align_face = cv2.resize(align_face, size)
                     return align_face                                  
                 else:
                      return None  
             else:
                return None  
    except  NameError:
        print(NameError)
        return None    
    
def oneAUG(img,batch_size=10,maxAug=5):
    gen = ImageDataGenerator(
                         rescale=1./255, 
                         width_shift_range = 0.4,
                         height_shift_range = 0.3,
                         zoom_range = 0.2,  ##0.2
                         rotation_range = 10,  ##10
                         shear_range=0.3,
                        )
    
    imgListAug=[]
    imgOri  = np.copy(img)
    imgFace = getFaceAug(img)
    
    if imgFace is not None: 
     
        imgOri         = imgOri.reshape((1,) + imgOri.shape)        
        cnt=0          
        pertama=True
        for X_batch in gen.flow(imgOri, batch_size=batch_size):
            imgAug  = np.array(X_batch*255).astype(np.uint8)       
            imgAug  = imgAug.reshape(img.shape)
            #plt.imshow(imgAug)
            #plt.show()
            imgFace = getFaceAug(imgAug)
            if imgFace is None:
                # break
                continue     
            print("agumentasi ke %d "%cnt)
            imgFace        = imgFace.reshape((1,) + imgFace.shape)        
            if pertama:
                imgListAug = imgFace
                pertama=False
            else:
                imgListAug = np.concatenate((imgListAug,imgFace),axis=0)
            cnt=cnt+1
            # imgEH   = np.array(exposure.equalize_hist(imgAug.flatten(),256)*255).astype(np.uint8)                               
            # imgEH   = imgEH.reshape(imgAug.shape[0],imgAug.shape[1],imgAug.shape[2])
            # imgFace  = getFaceAug(imgEH)            
            # if imgFace is not None:
            #     if not pertama:
            #         imgFace        = imgFace.reshape((1,) + imgFace.shape)        
            #         imgListAug = np.concatenate((imgListAug,imgFace),axis=0)
            #         cnt=cnt+1
            
            # imgEH   = np.array(exposure.equalize_adapthist(imgAug)*255).astype(np.uint8)                               
            # imgFace   = getFaceAug(imgEH)                        
            # if imgFace is not None:
            #     if not pertama:
            #         imgFace        = imgFace.reshape((1,) + imgFace.shape)        
            #         imgListAug = np.concatenate((imgListAug,imgFace),axis=0)
            #         cnt=cnt+1
            # p2, p98 = np.percentile(imgAug, (2, 98))
            # imgEH   = exposure.rescale_intensity(imgAug, in_range=(p2, p98))
            # imgFace = getFaceAug(imgEH)            
            # if imgFace is not None:
            #     if not pertama:
            #         imgFace        = imgFace.reshape((1,) + imgFace.shape)        
            #         imgListAug = np.concatenate((imgListAug,imgFace),axis=0)
            #         cnt=cnt+1                                            

            if cnt >= maxAug:                
                break                         
    return imgListAug

def plot_results(path, history):
       """
       Plot accuracy and loss trendency graph
        
       input:
       1. path - str: graph path
       2. history: model trained history
       """
       acc_name = 'accuracy_simpler.png'
       loss_name = 'loss_simpler.png'
       # Accuracy learning curves
       plt.figure(0)
       plt.plot(history.history['accuracy'])
       plt.plot(history.history['val_accuracy'])
       plt.title('model accuracy')
       plt.ylabel('accuracy')
       plt.xlabel('epoch')
       plt.legend(['train', 'val'], loc='upper left')
       plt.savefig(path+acc_name)
       plt.close()

        # Loss learning curves
       plt.figure(1)
       plt.plot(history.history['loss'])
       plt.plot(history.history['val_loss'])
       plt.title('model loss')
       plt.ylabel('loss')
       plt.xlabel('epoch')
       plt.legend(['train', 'val'], loc='upper left')
       plt.savefig(path+loss_name)
       plt.close()
        

if __name__ == '__main__':         
    
    fileFitur = 'faceRecognotionOK3.npz'
    path = "lfwTest" 
    nmFileAug = "faceRecognotionAug3.npz"
    if os.path.exists(nmFileAug): 
        dataLoaded = np.load(nmFileAug)       
        x_train  =dataLoaded['x_train']
        y_train  =dataLoaded['y_train']
        x_val    =dataLoaded['x_val']
        y_val    =dataLoaded['y_val']
        num_classes=dataLoaded['num_classes']
        print("Data loaded ...")
    else:
        detector = mtcnn.MTCNN()        
        if os.path.exists(fileFitur): 
           print("File %s already exist"%fileFitur)           
           dataLoaded = np.load(fileFitur)       
           ftr  =dataLoaded['ftr']
           # lbl  =dataLoaded['lbl']
           nm   =dataLoaded['nm']
           # usia =dataLoaded['usia']        
           print("loaded success")  
        else:              
            # train = pd.read_csv("train1.csv")
            # ftrs,lbls,nms,usias = [],[],[],[]   
            ftrs,nms = [],[],[],[]
            
            # ftrs,lbls,nms,usias = list(ftr),list(lbl),list(nm),list(usia)
            # cnt=0            
            for i in range(len(train)):
                # nmPath = path + "/%d/"%(i+1)
                nmPath = path
                # jk  =train['jenis kelamin'][i]
                # umur=train['usia'][i]
                # ftr,lbl,usia,nm = loadImageDetectFace(nmPath,jk,umur) 
                ftr, nm = loadImageDetectFace(nmPath) 
                numFaceDetected = len(list(set(nm)))
                if numFaceDetected !=3:
                     print("tidak lengkap 3 face dalam 1 folder %d, try with fr"%numFaceDetected)
                     # ftr,lbl,usia,nm = loadImageDetectFace(nmPath,jk,umur,fr=True)     
                     ftr, nm = loadImageDetectFace(nmPath,fr=True)     
                     numFaceDetected = len(list(set(nm)))
                     if numFaceDetected !=3:
                         print("tidak lengkap 3 face dalam 1 folder %d "%(numFaceDetected))
                
                if numFaceDetected ==3:
                     ftrs+=ftr
                     # lbls+=lbl
                     nms+=nm
                     # usias+=usia
                     # cnt+=1
                print("############   %d"%i)
                # if i>=9:
                  # break
            np.savez_compressed(fileFitur,ftr=np.array(ftrs),
                              # lbl=np.array(lbls),
                              nm=np.array(nms),
                              # usia=np.array(usias))
                              )
         
            # for i in range(len(lbls)):
            #  folder = "resultVGG_FR_New/%s"%nms[i].split("_")[0]
            #  if not os.path.exists(folder):
            #      os.makedirs(folder)
            #  nmFile = "%s/%s"%(folder,nms[i])    
            #  plt.imsave(nmFile,ftrs[i])
        
        ftr = getResize(ftr)
        ftr=np.asarray(ftr).astype(np.float32)    
        
        print("splitting traing and validation")
        print("-------------------------------")
        
        
        train,val,idImgs,addImgs,tmp_val = [],[],[],[],[]
        pertama=True
        batch_size=100
        maxAug=10
        
        for i in range(len(ftr)):     
        # for i in range(6,12):     
            idImg = int(nm[i].split("_")[0])
            idImgs.append(idImg)                
            nmFile = path + "/" + nm[i].split("_")[0] + "/" + nm[i]
            print("nmFile ...." + nmFile)
            img = plt.imread(nmFile) 
            imgListAug = oneAUG(img,batch_size,maxAug)
            if len(imgListAug)!=0:
                val.append(i)
                # if nm[i][-5:]=='1.jpg':
                #      tmp_val.append(i)
                #      enter=False
                # elif nm[i][-5:]=='2.jpg':
                #      train.append(i)   
                #      if len(tmp_val)>0:
                #          val = val+tmp_val
                #          tmp_val=[]
                #      enter=True
                # elif nm[i][-5:]=='3.jpg':
                #      train.append(i)  
                #      if len(tmp_val)>0:
                #          val = val+tmp_val
                #          tmp_val=[]                     
                #      enter=True
                # if enter:
                if pertama:
                    addAug = imgListAug
                    pertama=False
                else:                        
                    addAug = np.concatenate((addAug,imgListAug),axis=0)
                for c in range(len(imgListAug)):               
                    addImgs.append(idImg)    
            if i>62:
                break
        idImgs     = np.array(idImgs).astype(int)
        num_classes = len(np.unique(idImgs))
        le        = LabelEncoder()
        idlbl     = le.fit_transform(idImgs)
        addImgs   = le.transform(addImgs)
        output = open('le.pkl', 'wb')
        pickle.dump(le, output)
        output.close()
        
        
        category     = to_categorical(idlbl,num_classes)
        add_category = to_categorical(addImgs,num_classes)
        
        # x_train = ftr[train]
        x_val   = ftr[val]
        # y_train = category[train]
        y_val   = category[val] 
        # x_train = np.concatenate((x_train, addAug),axis=0) 
        # y_train = np.concatenate((y_train, add_category),axis=0) 
        x_train = addAug
        y_train = add_category
        
        
        print("check pasangan train-val, jk val tidak ada, hapus di train")
        yy_train = np.argmax(y_train,axis=1)
        yy_val   = np.argmax(y_val,axis=1)
        for c in range(num_classes):
            # if c==9:
                # print(y_train.shape)
                # break  
            idx_val = np.where(yy_val==c)[0]
            if len(idx_val)==0:
                print("delete train...%d"%c)
                idx_train = np.where(yy_train==c)[0]
                y_train=np.delete(y_train,idx_train,axis=0)
                x_train=np.delete(x_train,idx_train,axis=0)  
                yy_train=np.delete(yy_train,idx_train,axis=0)  
            # if c==9:
                # print(y_train.shape)
                # break  
            
                                      
            idx_train = np.where(yy_train==c)[0]
            if len(idx_train)<=5:
                print("delete val ...%d"%c)
                idx_val = np.where(yy_val==c)[0]
                y_val=np.delete(y_val,idx_val,axis=0)
                x_val=np.delete(x_val,idx_val,axis=0) 
                yy_val=np.delete(yy_val,idx_val,axis=0)  
                
                y_train=np.delete(y_train,idx_train,axis=0)
                x_train=np.delete(x_train,idx_train,axis=0)                    
                yy_train=np.delete(yy_train,idx_train,axis=0)  
                
                     
            
        np.savez_compressed(nmFileAug,
                                x_train=x_train,y_train=y_train,
                                x_val=x_val,y_val=y_val,
                                num_classes=num_classes
                                )     
        
    


    print(x_train.shape)    
    print(x_val.shape)   
    needSave=False
    if needSave:
        for i in range(len(x_train)):
            folder = "resultAug/Train/%s"%np.argmax(y_train[i])
            if not os.path.exists(folder):
                os.makedirs(folder)
            nmFile = "%s/%d.png"%(folder,i)    
            plt.imsave(nmFile,x_train[i].astype(np.uint8))

        for i in range(len(x_val)):
            folder = "resultAug/Test/%s"%np.argmax(y_val[i])
            if not os.path.exists(folder):
                os.makedirs(folder)
            nmFile = "%s/%d.png"%(folder,i)    
            plt.imsave(nmFile,x_val[i].astype(np.uint8))
    
    train_datagen = ImageDataGenerator(
                                        rescale=1./255, 
                                        # width_shift_range = 0.5,
                                        # height_shift_range = 0.5,
                                        # zoom_range = 0.5,  ##0.2
                                        # rotation_range = 30,  ##10
                                        # shear_range=0.5,                                                             
                                        # horizontal_flip=True, 
                                        # fill_mode='nearest',  
                                        # featurewise_center=True,
                                        # samplewise_center=True,
                                        # featurewise_std_normalization=True,
                                        # brightness_range = [0.0,0.1]                                                                              
                                        )  
    # train_datagen.fit(x_train)
    val_datagen = ImageDataGenerator(rescale=1./255)

    batch_size =32
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)
    nmModel   = 'simplerCNN.h5'
    callbacks = get_callbacks(nmModel, patience_lr=10)
    input_shape = (DIM, DIM, RGB)
    clear_session()
    # model = SmallerVGGNet.build(width=DIM, height=DIM, depth=RGB,
    #                         classes=num_classes)
    model = simpler_CNN(input_shape, num_classes)
    # model = alexNet(input_shape, num_classes)        
    # model = simpleCNN(input_shape, num_classes)        

    # model.compile(optimizer='adam',
    #            loss='categorical_crossentropy',
    #            metrics=['accuracy'])
    # print(model.summary())  
    # base_model = InceptionResNetV2(include_top=False, pooling='avg',
    #                                input_shape=input_shape,weights=None)
       
    
    
    # base_model.trainable = True
    # print(base_model.summary())
    # x = base_model.output
    # x = Flatten()(x)
    # x = Dense(1024, activation="relu")(x)
    # x = Dropout(0.4)(x)
    # x = Dense(1024, activation="relu")(x)
    # x = Dropout(0.4)(x)
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.1)(x)
    
    
    # predictions = Dense(num_classes, activation="softmax")(x)    
    

    adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
    # model = Model(base_model.input, predictions)
    model.compile(loss ='categorical_crossentropy', #tf.keras.losses.LogCosh(),
                  # tf.keras.losses.Huber(delta=1.5),
                  optimizer=adam, 
                  metrics=['accuracy']                  
                  )     

    num_data = int(len(y_train))
    epochs   = 10
    sum_data = int(num_data/batch_size)
    needTrain=True
    if needTrain:
        history = model.fit(
                train_generator,
                steps_per_epoch=sum_data,
                epochs=epochs,                
                validation_data=val_generator,
                # validation_steps=sum_data,
                callbacks=[callbacks],verbose=1,
                validation_steps=int(len(y_val) / batch_size)
                )
  
        plot_results("", history)
        model.load_weights(nmModel) 
        print("Validation acc")
        scores = model.evaluate(x_val/255,y_val, verbose=0)    
        print(scores)
   
    

