import numpy as np
import cv2
img=cv2.imread(r"tree.jpg",1)
cv2.circle(img,(80,80),55,(0,255,0),-1)
cv2.imshow('image',img)
cv2.waitkey(0)
#cv2.destroyAllWindows()
img=cv2.imread(r"face.jpg",1)
cv2.rectangle(img,(15,25),(200,150),(0,255,255),5)
cv2.imshow('image',img)
#cv2.waitkey(0)
#cv2.destroyAllWindows()
img=cv2.imread(r"vehicle.jpg",1)
cv2.ellipse(img,(150,50),(80,20),5,0,360,(0,0,255),-1)
cv2.imshow('image',img)
#cv2.waitkey(0)
#cv2.destroyAllWindows()
import numpy as np
import cv2
