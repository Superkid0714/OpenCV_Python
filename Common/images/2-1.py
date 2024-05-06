import cv2

image = cv2.imread("mars.jpeg",cv2.IMREAD_COLOR)
if image is None: raise Exception ("오류발생")

x_axis = cv2.flip(image,0)
y_axis = cv2.flip(image,1)
xy_axis = cv2.flip(image,-1)
rap_image = cv2.repeat(image,1,2)
trans_image = cv2.transpose(image)

titles =['image','x_axis','y_axis','xy_axis','rap_image','trans_image' ]
for title in titles:
    cv2.imshow(title,eval(title))
cv2.waitKey(0)
