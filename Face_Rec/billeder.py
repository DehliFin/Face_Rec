import cv2
import sys

cascPath = "cascades\data\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(r"Images\abba.png")

# Used to scale the image if it too big for the screen
#scale_percent = 20
#dim = (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100))
#resize = cv2.resize(image, dim)



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the resulting frame
cv2.imshow("faces found", image)

# Can be used to save the image
#cv2.imwrite("", image)

cv2.waitKey(0)

cv2.destroyAllWindows()
