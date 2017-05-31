
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

outputDir = "/Volumes/MyBook/"
def writeToFile(content):
	with open(outputDir + 'dummy_labels.txt', 'a') as f:
	# with open('dummy_labels.txt', 'a') as f:
		f.write(content)

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
	if counter%100 == 0:
		print "counter: ", counter

encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
print encoded_labels
dummy_labels = np_utils.to_categorical(encoded_labels)
print dummy_labels.shape
print dummy_labels[0]
#
batch = 100
maxIter = int(len(dummy_labels)/float(batch))
print "maxIter: ", maxIter

for i in range(maxIter):
	data = dummy_labels[i * batch : (i + 1) * batch]
	print "batch num: ", i
	print "data.shape: ", data.shape

	content = ""
	for line in data:
		# print "type(line): ", type(line)
		for num in line:
			content += (str(int(num)) + " ")
		content += "\n"
	writeToFile(content)

print "last batch"
data = dummy_labels[(i + 1) * batch :]
print "data.shape: ", data.shape
if len(data) != 0:
	print "write for last batch"
	content = ""
	for line in data:
		# print "type(line): ", type(line)
		for num in line:
			content += (str(int(num)) + " ")
		content += "\n"
	writeToFile(content)

# np.save(outputDir + 'dummy_labels.npy', dummy_labels)
# pickle.dump( dummy_labels, open( "dummy_labels.p", "wb" ) )

# with open('dummy_labels.txt', 'r') as f:
# 	lines = f.readlines()
# for line in lines:
# 	print line
# 	break


# with open('dummy_labels.txt', 'r') as f:
# 	lines = f.readlines()

# for line in lines:
# 	line = line.replace(" \n", "")
# 	print map(int, line.split(" "))

# with open('MScleanTrainFilterNovelBase.txt', 'r') as f:
# 	lines = f.readlines()
# print len(lines)
