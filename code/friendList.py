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
	outputFriendFile = parentDir + args.inputFriendFile + '_new.csv'
	node2vecFriendFile = parentDir + args.inputFriendFile + '_new.edgelist'
	threshold = int(args.threshold)

	userReviewCount = Counter()
	with open(reviewFile, 'r') as reviewFileReader:
		skipLine = reviewFileReader.readline()
		for line in reviewFileReader:
			line = line.replace("\n","").split(",")		
			userReviewCount[int(line[0])] += 1


	inputFriendFileReader = open(inputFriendFile, 'r')
	skipLine = inputFriendFileReader.readline()
	node2vecFriendFileWriter = open(node2vecFriendFile, 'w')
	outputFriendFileWriter = open(outputFriendFile, 'w')
	outputFriendFileWriter.write('userID,numFriends,friendList\n')
	for line in inputFriendFileReader:
		line = line.replace("\n","").split(",")
		userID = int(line[0])
		numFriend = int(line[1])
		if numFriend == 0:
			outputFriendFileWriter.write(str(userID) + ',' + str(0) + ',' + '\n')
			continue
		# print(line[0], len(line[2]))
		friendList = [int(f) for f in line[2].split("::")]
		friendListNew = []
		for friend in friendList:
			if userReviewCount[friend] > threshold:
				# if userReviewCount[friend] == 0:
					# print('!!no review: ' + str(userID))
				friendListNew.append(str(friend))
				node2vecFriendFileWriter.write(str(userID) + ' ' + str(friend) + '\n')
			# else:
			# 	print('no review: ' + str(userID))
		numFriendNew = len(friendListNew)
		lineNew = str(userID) + ',' + str(numFriendNew) + ',' + '::'.join(friendListNew) + '\n'
		outputFriendFileWriter.write(lineNew)

	inputFriendFileReader.close()
	node2vecFriendFileWriter.close()
	outputFriendFileWriter.close()
