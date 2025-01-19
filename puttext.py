import numpy as np
import cv2
font=cv2.FONT_HERSHEY_SIMPLEX
img=cv2.imread(r"face.jpg",1)
cv2.putText(img,'face',(20,45),font,2,(0,200,140),5)
cv2.imshow("image",img)
cv2.waitKey(0)
