# camera.py
###import f_Face_info
import cv2
import PIL.Image
from PIL import Image
import time
import imutils
import argparse
import shutil
#import pytesseract
import imagehash
import json
import PIL.Image
from PIL import Image
from PIL import ImageTk
from random import randint

#from deepface import DeepFace


import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="ai_consultancy"
)


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        #Live Video Capture
        self.video = cv2.VideoCapture(0)
        ##FR
        self.video.set(3, 640) # set video widht
        self.video.set(4, 480) # set video height

        # Define min window size to be recognized as a face
        self.minW = 0.1*self.video.get(3)
        self.minH = 0.1*self.video.get(4)
        ##
        self.k=1
        #cap = self.video
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('video.mp4')

        # Check if camera opened successfully
        #if (cap.isOpened() == False): 
        #  print("Unable to read camera feed")

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        #frame_width = int(cap.get(3))
        #frame_height = int(cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #self.out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


        
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, image = self.video.read()
        #self.out.write(image)
        try:
            cv2.imwrite("getimg.jpg", image)
            
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # Read the frame
            #_, img = cap.read()

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            ff=open("user1.txt","r")
            uuid=ff.read()
            ff.close()
            
            #Feature Extraction-Local Binary Patterns  (LBP)
            ###FR
            id = 0
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            tn="trainer"+uuid+".yml"
            #print("test")
            #print(tn)
            recognizer.read('trainer/'+tn)
            cascadePath = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(cascadePath);

            font = cv2.FONT_HERSHEY_SIMPLEX
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(self.minW), int(self.minH)),
               )
            
            # Draw the rectangle around each face
            j = 1

            ff=open("user.txt","r")
            uu=ff.read()
            ff.close()

            

            ff1=open("photo.txt","r")
            uu1=ff1.read()
            ff1.close()
            
            ff1=open("emotion.txt","r")
            em1=ff1.read()
            ff1.close()
            
            ###########################################
            cursor = mydb.cursor()
            #Frame Extraction        
            j=1
            for (x, y, w, h) in faces:
                mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imwrite("static/myface.jpg", mm)

                if em1=="":
                    image = cv2.imread("static/myface.jpg")
                    cropped = image[y:y+h, x:x+w]
                    gg="f"+str(j)+".jpg"
                    cv2.imwrite("static/faces/"+gg, cropped)
                ##FR
                #try:
                cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                uid=id
                g2=0
                g1=""
                fdata=[]
                name=""
                emo=""
                if uid=="":
                    s=1
                else:
                    
                    if id>100:
                        g1="c"
                        g2=id-100
                    elif id>50:
                        g1="b"
                        g2=id-50
                    else:
                        g1="a"
                        g2=id
                
                    
                    
                    cursor.execute('SELECT count(*) FROM ac_user where id=%s',(g2,))
                    cnn = cursor.fetchone()[0]
                    
                    
                    if cnn>0:
                        cursor.execute('SELECT * FROM ac_user where id=%s',(g2,))
                        fdata = cursor.fetchone()
                        if id>100:
                            emo="Severe"
                            name=fdata[1]
                        elif id>50:
                            emo="Moderate"
                            name=fdata[1]
                        else:
                            emo="Mild"
                            name=fdata[1]
                            
                        
                    else:
                        ff=open("facest.txt","w")
                        ff.write("no")
                        ff.close()
                        

                # Check if confidence is less them 100 ==> "0" is perfect match 
                if (confidence < 50):
                    id = name
                    #namex[id]
                    ff=open("facest.txt","w")
                    ff.write(str(uid))
                    ff.close()

                    ff=open("emotion.txt","w")
                    ff.write(emo)
                    ff.close()
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = ""
                    ff=open("facest.txt","w")
                    ff.write("no")
                    ff.close()

                    
                    confidence = "  {0}%".format(round(100 - confidence))
                if em1=="":
                    cv2.putText(image, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                    cv2.putText(image, emo, (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
                #except:
                #    print("try")
                ##
                #mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 200, 0), 1)
                #cv2.imwrite("static/myface.jpg", mm)

                #image1 = cv2.imread("static/myface.jpg")
                #cropped = image1[y:y+h, x:x+w]
                #gg="f"+str(j)+".jpg"
                #cv2.imwrite("static/faces/"+gg, cropped)

                j+=1

           
            ##########################
            parser1 = argparse.ArgumentParser(description="Face Info")
            parser1.add_argument('--input', type=str, default= 'webcam',
                                help="webcam or image")
            parser1.add_argument('--path_im', type=str,
                                help="path of image")
            args1 = vars(parser1.parse_args())

            type_input1 = args1['input']
        except:
            print("try")
        
            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
