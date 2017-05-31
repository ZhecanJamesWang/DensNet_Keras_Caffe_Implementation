import cv2
import utility as ut

# generateFunc = ["original", "scale", "rotate", "translate", "scaleAndTranslate", "brightnessAndContrast"]


img = cv2.imread("testImg/testImg.jpg", 1)
print img.shape
# newImg, x, y = ut.scale(img, [], [])
# newImg, x, y = ut.mirror(img, [], [])
# newImg, x, y = ut.contrastBrightess(img, [], [])
# newImg, x, y = ut.rotate(img, [], [])
# newImg, x, y = ut.translate(img, [], [])

print newImg.shape

cv2.imshow("img", img)
cv2.imshow("newImg", newImg)
cv2.waitKey(0)
