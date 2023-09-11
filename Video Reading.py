import cv2
import numpy as np
offset = 25
#ask name of person
file_name = input("Enter name of person: ")
dataset_path = "" # USER: Enter path of images dataset 

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier(r"") # USER: Enter path to save findings 
#create list to save cropped images
face_data = []
skip = 0

while True:
    flag, img = cam.read()
    if flag == False:
        print("Reading Camera Failed!")
    
    grey_image =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(img, 1.1, 10)
    faces = sorted(faces, key=lambda f:f[2]*f[3])
    #selecting largest face
    if len(faces) > 0:
        f = faces[-1]
    
    x,y,w,h = f
    cv2.rectangle(img, (x - offset,y - offset), (x+w+offset,y+h+offset), (0,255,0),2)
    
    #crop and save largest face
    cropped_face = img[y - offset:y+h+offset, x - offset:x+w+offset]
    #resizing image to standard
    cropped_face = cv2.resize(cropped_face, (100, 100))
    skip += 1
    if skip % 10 == 0:
        face_data.append(cropped_face)
        print("saved so far " + str(len(face_data)))
        
        
    cv2.imshow("Sunny", img)
    cv2.imshow("cropped image ", cropped_face)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
#write face data on hard disk
face_data = np.asarray(face_data)
print(face_data.shape)
n = face_data.shape[0]
face_data = face_data.reshape((n, -1))
print(face_data.shape)
np.save(dataset_path + file_name + ".npy", face_data)
cam.release()
cv2.destroyAllWindows()