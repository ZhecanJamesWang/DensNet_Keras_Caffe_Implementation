
import numpy as np
import pickle 

outputDir = "/Volumes/MyBook/"
# def writeToFile(content):
# 	with open(outputDir + 'dummy_labels.txt', 'a') as f:
# 	# with open('dummy_labels.txt', 'a') as f:
# 		f.write(content)

with open('MScleanTrainFilterNovelBase.txt', 'r') as f:
	lines = f.readlines()
print len(lines)
labels = []
counter = 0
for line in lines:
	split = line.split("\t")
	label = split[1].replace("\n", "")
	labels.append(label)
	counter += 1
	# if counter >= 130:
	# 	break
	if counter%1000  == 0:
		print "counter: ", counter
print "len(labels): ", len(labels)
lablesSet = set(labels)
labels = list(lablesSet)
print "len(labels): ", len(labels)

pickle.dump( labels, open( outputDir + "uniqueLabelList.p", "wb" ) )
		