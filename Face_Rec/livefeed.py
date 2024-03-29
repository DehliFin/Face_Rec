import cv2
import sys
#from Face_reg import frame
cascPath = "cascades\data\haarcascade_frontalface_alt2.xml"
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cascPath)
cap.set(3,1000)
cap.set(4,500)
while(True):    

    # Capture frame-by-frame
    ret, frame = cap.read()

    
    # Get user supplied values
    imagePath = "frame"

    

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(30, 30),
        
    )

    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w] # roi = reigon of interest
        roi_color = frame[y:y+h, x:x+w]

        print(x, y, w, h)

        color = (0, 255, 0)#Green BGR
        stroke = 2 #thickness on border
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
     
   
    # Display the resulting frame
   
    cv2.imshow('frame',frame)
    
    cv2.waitKey(41) #24 fps

cap.release()
cv2.destroyAllWindows()