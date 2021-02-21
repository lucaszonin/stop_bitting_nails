from app import PreparingModel
import cv2 as cv
import numpy as np


class PredictingCam(object):

    def __init__(self):

        self.file_name = "haarcascade_frontalface_alt2.xml"
        self.classifier = cv.CascadeClassifier(f"{cv.haarcascades}/{self.file_name}")
        self.camera = cv.VideoCapture(0)
        self.X_train, self.X_test, self.y_train, self.y_test = PreparingModel().loading_dataframe()
        self.main()


    def main(self):
        
        label = {
            0 : "Não roendo",
            1 : "Roendo"
        }

        #self.X_train = PreparingModel().pca_model(self.X_train)
        #self.X_test = PreparingModel().pca_model(self.X_test)  

        model = PreparingModel().knn.fit(self.X_train, self.y_train) 

        while True:

            status, frame = self.camera.read()

            if not status:
                break

            if cv.waitKey(1) & 0xff == ord('q'):
                break
            
            to_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                
            faces = self.classifier.detectMultiScale(to_gray)


            for x,y,w,h in faces:

                gray_face = to_gray[y:y+h, x:x+w]

                #vector = self.pca.transform([gray_face.flatten()])

                if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:

                    gray_face = cv.resize(gray_face, (150,150))

                    predicted = model.predict(gray_face)[0] 

                    classification = label[predicted]

                    if predicted == 0:
                        cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
                        print("\a")
                    elif predicted == 1:
                        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
                    
                    cv.putText(frame, classification, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)
                    cv.putText(frame, f"{len(faces)} rostos identificados",(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

            cv.imshow("Camera", frame)

if __name__ == "__main__":
    d = PredictingCam()    