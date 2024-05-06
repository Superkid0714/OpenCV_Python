import cv2
import numpy as np
import matplotlib.pyplot as plt

sj_image  = cv2.imread("sj.jpg", cv2.IMREAD_COLOR)
background_image = cv2.imread("background.jpg")
if sj_image is None: raise Exception("영상 파일 읽기 에러")
if background_image is None: raise Exception("영상 파일 읽기 에러")

sj_image_resized = cv2.resize(sj_image, (background_image.shape[1], background_image.shape[0]))

alpha_channel = cv2.cvtColor(sj_image_resized, cv2.COLOR_BGR2GRAY)
alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)

alpha_channel_normalized = np.zeros_like(alpha_channel, dtype=np.float32)
cv2.normalize(alpha_channel, alpha_channel_normalized, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

alpha_channel_normalized = (alpha_channel_normalized * 255).astype(np.uint8)

foreground_mask = cv2.bitwise_and(sj_image_resized, sj_image_resized, mask=alpha_channel_normalized)
background_mask = cv2.bitwise_and(background_image, background_image, mask=1 - alpha_channel_normalized)
merge_image = cv2.add(foreground_mask, background_mask)

rows, cols = sj_image.shape[:2]
rgb_img = cv2.cvtColor(sj_image, cv2.COLOR_BGR2RGB)

fig = plt.figure(num=1, figsize=(3, 4))
plt.imshow(merge_image), plt.title('If you do not walk today, you will have to run tomorrow')
plt.axis('off'), plt.tight_layout()
plt.show()