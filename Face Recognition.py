import warnings
import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

data_path = "" # USER: Enter path for Dataset
face_data = []
labels = []
class_id = 0
name_map = {}
offset = 10

for f in os.listdir(data_path):
    if f.endswith(".npy"):
        name_map[class_id] = f[:-4]
        #x values
        dataItem = np.load(data_path + f)
        m = dataItem.shape[0]
        face_data.append(dataItem)
        #y values
        target = class_id * np.ones((m,))
        class_id +=1
        labels.append(target)
XT = np.concatenate(face_data, axis=0)
yT = np.concatenate(labels, axis=0).reshape((-1,1))
print(XT.shape)
print(yT.shape)
print(name_map)

cam = cv2.VideoCapture(0)
# Requires the download of OpenCV Haar cascades (Available in the repository) 
model = cv2.CascadeClassifier(r"") # USER: Enter path of Haar cascades

while True:
    flag, img = cam.read()
    if flag == False:
        print("Reading Camera Failed!")
    
    faces = model.detectMultiScale(img, 1.1, 10)
    for f in faces:
   
        x,y,w,h = f

        #crop and save largest face
        cropped_face = img[y - offset:y+h+offset, x - offset:x+w+offset]
        #resizing image to standard
        cropped_face = cv2.resize(cropped_face, (100, 100))


        cropped_face = cropped_face.flatten()
        cropped_face = cropped_face.reshape(1, -1)
        #prediction using KNN algorithm
        k = 5  
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(XT, yT)
        y_pred = knn_classifier.predict(cropped_face)
        name_predicted = name_map[y_pred[0]]
        #Display the name and Box
        cv2.rectangle(img, (x - offset,y - offset), (x+w+offset,y+h+offset), (0,255,0),2)
        cv2.putText(img, name_predicted, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        warnings.filterwarnings("ignore")
    
    cv2.imshow("prediction_window ", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()