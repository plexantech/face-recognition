import os 
import cv2
# import pygame
import pyttsx3
import numpy as np
import face_recognition


class FaceRecognition:
    """
    Class for monitoring class attendance using face recognition.

    Attributes:
        IMAGE_PATH (str): The path to the directory containing student images.
        img_data (list): A list of dictionaries containing student data.
    """
    
    def __init__(self): 
        """Initialize the class."""
        self.IMAGE_PATH = "images"
        self.img_data = [
            {"name": "Elon Musk", "image": "elon_musk.jpg"}, 
            {"name": "Jeff Bezos", "image": "jeff_bezos.jpg"}, 
            {"name": "Jack Ma", "image": "jack_ma.jpg"}, 
            {"name": "Mark Zuckerburg", "image": "jack_ma.jpg"},
            {"name": "Ismael Swaleh", "image": "ismael_swaleh.jpg"},
            # {"name": "Magomu Rayan", "image": "rayan.jpg"}
        ]
        
        
    def findEncodings(self): 
        """
        Find face encodings for known students.

        Returns:
            list: A list of face encodings for known students.
        """
        encodeList = []
        for data in self.img_data:
            data_img = os.path.join(self.IMAGE_PATH, data["image"])
            img = face_recognition.load_image_file(data_img)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            
        return encodeList   
    
    
    def markAttendance(self, name):
        """
        Mark attendance for the recognized student.

        Args:
            name (str): The name of the recognized student.
        """
        
        engine = pyttsx3.init()
        
        # Fire Up the sound
        if name != "Unknown": 
            engine.say(f"Welcome {name}")
            engine.runAndWait()
        
        else: 
            engine.say(name)
            engine.runAndWait()
            
            
        print(f"Attendance marked for {name}")

    
    def main(self): 
        """Main function to run the attendance monitoring software."""
        self.classNames = [data["name"] for data in self.img_data]  # Extract names
        encodeListKnown = self.findEncodings()
        
        cap = cv2.VideoCapture(0)
        
        print("started")
        
        while True: 
            _, img = cap.read()
            img = cv2.flip(img, 1)
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):         
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)
                
                if matches[matchIndex]: 
                    name = self.classNames[matchIndex].capitalize()
                    self.markAttendance(name)  # Call attendance marking function
                    
                else:
                    name = "Unknown"  # Mark as unknown for unrecognized faces
                    
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Class Attendance", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
if __name__ == "__main__": 
    system = FaceRecognition()
    system.main()
