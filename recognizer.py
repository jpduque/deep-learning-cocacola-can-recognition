import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from CompareImage import CompareImage
import progressbar
import time

cap = cv2.VideoCapture(0)
t_end = time.time() + 3
index = 0
while time.time() < t_end:
    ret, frame = cap.read()
    name = 'images/frame' + str(index) + '.jpg'
    cv2.imwrite(name, frame)
    index = index + 1
cap.release()
cv2.destroyAllWindows()

templates = [file for file in glob.glob("template/*.jpg")]
templates = templates[:15]
videos = [file for file in glob.glob("images/frame*.jpg")]

assertion = []

bar = progressbar.ProgressBar(maxval=len(videos),
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for idx, vid in enumerate(videos):
    scoring = []
    for file in templates:
        compare_image = CompareImage(file, vid)
        image_difference = compare_image.compare_image()
        scoring.append(image_difference)
    bar.update(idx)
    assertion.append(np.mean(scoring))

bar.finish()
similarity_score = np.mean(assertion)

print(similarity_score)

s_img = cv2.imread("template/frame0.jpg")
img = 0
if similarity_score <= 0.4:
    img = cv2.imread("approved.png")
else:
    img = cv2.imread("rejected.png")

files = glob.glob('images/*')
for f in files:
    os.remove(f)

plt.imshow(img)
plt.show()
