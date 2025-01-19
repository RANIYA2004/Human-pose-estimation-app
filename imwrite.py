import cv2
img=cv2.imread(r'tree.jpg',0)
print(img)
status=cv2.imwrite('tree.jpg',img)
print("Image written to file-system : ",status)