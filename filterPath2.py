# /Volumes/usb1/ms2/
#
# /media/usbstick/MS_challenge1_train_data
#
#
def writeToFile(content):
	with open('MScleanTrainFilterNovelBaseFilterLowImageIdentityServer.txt', 'a') as f:
		f.write(content)


with open('MScleanTrainFilterNovelBaseFilterLowImageIdentity.txt', 'r') as f:
	lines = f.readlines()

counter = 0
content = ""
for line in lines:
    if "MS_challenge1_train_data" in line:
    	line = line.replace("/Users/zhecanwang/Project/MS-Celeb-1M/baseImage/", "/home/james/MS-Celeb-1M/baseImage_224/")
    	# line = line.replace('.jpg', '.png')
    	content += line
    	counter += 1
    	if counter % 100 == 0:
    		writeToFile(content)
    		content = ""
    		print counter



try:
  if "MS_challenge1_train_data" in line:
    # newLine = line
    # newLine = newLine.replace("MS_challenge1_train_data", "MS_challenge1_train_data_224")
    filePath = line[len('/Volumes/MyBook/MS_challenge1_train_data/'):].split('\t')[0].replace(".jpg", ".png")
    # folderPath = "/Volumes/MyBook/MS_challenge1_train_data_224/"
    folderPath = "/media/usbstick/MS_challenge1_train_data_224/"

    # print line.split('\t')[0]
    if os.path.exists('/media/usbstick/MS_challenge1_train_data_224/' + filePath) == False:
      # img = cv2.imread(line.split('\t')[0].replace(".jpg", ".png"), 1)
      img = cv2.imread('/media/usbstick/MS_challenge1_train_data/' + filePath, 1)
      if img != None:
        # print "img.shape: ", img.shape
        newImg, x, y = ut.scale(img, [], [], imSize = 224)
        # print "newImg.shape: ", newImg.shape
        if os.path.exists(folderPath + filePath.split("/")[0]) == False:
          os.mkdir(folderPath + filePath.split("/")[0])
          print "create dir: " + folderPath + filePath.split("/")[0]
        cv2.imwrite(folderPath + filePath, newImg)
        print "******** write to : ", folderPath + filePath
    else:
      print "skip: " + '/media/usbstick/MS_challenge1_train_data/' + filePath

  elif "MS_challenge2_baseset_data" in line:
    print line
    # newLine = line
    # newLine = newLine.replace("MS_challenge1_train_data", "MS_challenge1_train_data_224")
    filePath = line[len('/Volumes/MyBook/MS_challenge2_baseset_data/'):].split('\t')[0]
    folderPath = "/media/usbstick/MS_challenge2_baseset_data_224/"
    print line.split('\t')[0]
    if os.path.exists('/media/usbstick/MS_challenge2_baseset_data_224/' + filePath.replace(".png", ".jpg")) == False:
      # img = cv2.imread(line.split('\t')[0].replace(".jpg", ".png"), 1)
      img = cv2.imread('/media/usbstick/MS_challenge2_baseset_data/' + filePath, 1)
      if img != None:
        print "img.shape: ", img.shape
        newImg, x, y = ut.scale(img, [], [], imSize = 224)
        print "newImg.shape: ", newImg.shape
        if os.path.exists(folderPath + filePath.split("/")[0]) == False:
          os.mkdir(folderPath + filePath.split("/")[0])
          print "create dir: " + folderPath + filePath.split("/")[0]
        cv2.imwrite(folderPath + filePath.replace(".jpg", ".png"), newImg)
        print "******** write to : ", folderPath + filePath
    else:
      print "skip: " + '/media/usbstick/MS_challenge2_baseset_data/' + filePath
  else:
    print "doesnt exist folder?????????????"
except Exception as e:
  print "e: ", e
  pass
