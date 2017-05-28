import os
import argparse
from collections import Counter

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('reviewFile')
	parser.add_argument('inputFriendFile')
	parser.add_argument('threshold')

	parentDir = "/Users/FYang/Documents/A_Study/D_Courses/C_2017Spr/CS341/tune_params/Yelp/data/"
	args = parser.parse_args()
	reviewFile = parentDir + args.reviewFile + '.csv'
	inputFriendFile = parentDir + args.inputFriendFile + '.csv'
	outputFriendFile = parentDir + args.inputFriendFile + '_top.csv'
	# node2vecFriendFile = parentDir + args.inputFriendFile + '_new.edgelist'
	threshold = int(args.threshold)

	userReviewCount = Counter()
	with open(reviewFile, 'r') as reviewFileReader:
		skipLine = reviewFileReader.readline()
		for line in reviewFileReader:
			line = line.replace("\n","").split(",")		
			userReviewCount[int(line[0])] += 1


	inputFriendFileReader = open(inputFriendFile, 'r')
	skipLine = inputFriendFileReader.readline()
	# node2vecFriendFileWriter = open(node2vecFriendFile, 'w')
	outputFriendFileWriter = open(outputFriendFile, 'w')
	outputFriendFileWriter.write('userID,numFriends,friendList\n')
	for line in inputFriendFileReader:
		line = line.replace('\n','').split(',')
		userID = int(line[0])
		numFriend = int(line[1])
		if numFriend <= threshold:
			outputFriendFileWriter.write(str(userID) + ',' + str(numFriend) + ',' + line[2] + '\n')
			continue
		friendList = [int(f) for f in line[2].split("::")]
		topFriendCount = {}
		for friend in friendList:
			topFriendCount[friend] = userReviewCount[friend]

		topFriendCount = sorted(topFriendCount, key=lambda x:topFriendCount[x])[0:threshold]
		topFriendCount = [str(k) for k in topFriendCount]
		lineNew = str(userID) + ',' + str(threshold) + ',' + '::'.join(topFriendCount) + '\n'
		outputFriendFileWriter.write(lineNew)

	inputFriendFileReader.close()
	outputFriendFileWriter.close()




