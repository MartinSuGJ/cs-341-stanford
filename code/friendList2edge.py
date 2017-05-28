import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='friend list to edges.')

	parser.add_argument('--friendList', nargs='?', default='data/yelp_user_friends_list_train.csv',
						help='Input friend path')

	parser.add_argument('--edgeList', nargs='?', default='data/yelp_user_friends_list_train.edgelist',
						help='output edge path')

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	friendList = args.friendList
	edgeList = args.edgeList

	inputFriendFileReader = open(friendList, 'r')
	outputFriendFileWriter = open(edgeList, 'w')
	skipLine = inputFriendFileReader.readline()
	# num = 0
	for line in inputFriendFileReader:
		line = line.replace("\n","").split(",")
		userID = line[0]
		numFriend = int(line[1])
		# num += numFriend
		if numFriend == 0:
			continue
		friends = line[2].split('::')
		for friend in friends:
			outputFriendFileWriter.write(userID + ' ' + friend + '\n')
	# print('num is: ' + str(num) + '\n')
	outputFriendFileWriter.close()
	inputFriendFileReader.close()


