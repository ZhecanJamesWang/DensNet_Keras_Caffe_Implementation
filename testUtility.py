import cv2
import utility as ut

# generateFunc = ["original", "scale", "rotate", "translate", "scaleAndTranslate", "brightnessAndContrast"]


img = cv2.imread("testImg/testImg.jpg", 1)
print img.shape

w, h, _ = img.shape
newImg, x, y = ut.scale(img, [], [])
print newImg.shape
newImg = cv2.resize(newImg, (h, w))
print newImg.shape

# newImg, x, y = ut.mirror(img, [], [])
# newImg, x, y = ut.contrastBrightess(img, [], [])
# newImg, x, y = ut.rotate(img, [], [])
# newImg, x, y = ut.translate(img, [], [])
# newImg, x, y = ut.resize(img, [], [])


cv2.imshow("img", img)
cv2.imshow("newImg", newImg)
cv2.waitKey(0)
