import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from CompareImage import CompareImage
import progressbar



templates = [file for file in glob.glob("images/*.jpg")]
vid = cv2.VideoCapture('videos/ccrotation.mp4')

index = 0
while (True):
    ret, frame = vid.read()
    if not ret:
        break
    name = 'images/frame' + str(index) + '.jpg'
    cv2.imwrite(name, frame)
    index += 1

videos = [file for file in glob.glob("images/frame*.jpg")]

assertion = []


bar = progressbar.ProgressBar(maxval=len(videos),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()


for idx, vid in enumerate(videos):
    scoring = []
    for file in templates:
        compare_image = CompareImage(file, vid)
        image_difference = compare_image.compare_image()
        scoring.append(image_difference)
        # print("hello")
    bar.update(idx)
    assertion.append(np.mean(scoring))

bar.finish()
similarity_score = 1 - np.mean(assertion)

print(similarity_score)


s_img = cv2.imread("images/frame0.jpg")
l_img = 0
if similarity_score >= 0.8:
    img = cv2.imread("approved.png")
    l_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
else:
    img = cv2.imread("rejected.png")
    l_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

x_offset = y_offset = 50
l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img


files = glob.glob('images/*')
for f in files:
    os.remove(f)


plt.imshow(l_img)
plt.show()
