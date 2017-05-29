	

import argparse

def parse_args():

	parser = argparse.ArgumentParser(description='Calculate valid friend review percentage')

	
	parser.add_argument('--trainAndVal', nargs='?', default='data/trainAndValidate.csv')
	parser.add_argument('--test', nargs='?', default='data/test.csv')
	parser.add_argument('--friendList', nargs='?', default='data/yelp_user_friends_list_trainAndValidate.csv')
	parser.add_argument('--output', nargs='?', default='data/test_reviewPecent.csv')
	return parser.parse_args()


def generatePercentage(args):
	trainAndVal = args.trainAndVal
	test = args.test
	friendList = args.friendList
	output = args.output

	reviews = {}
	trainAndValReader = open(trainAndVal, 'r')
	skipLine = trainAndValReader.readline()
	for line in trainAndValReader:
		line = line.replace('\n','').split(',')
		if (line[0], line[1]) not in reviews:
			reviews[(line[0], line[1])] = 1
	trainAndValReader.close()

	friends = {}
	friendListReader = open(friendList, 'r')
	skipLine = friendListReader.readline()
	for line in friendListReader:
		line = line.replace('\n', '').split(',')
		if int(line[1]) == 0:
			continue
		friends[line[0]] = line[2].split('::')


	outputWriter = open(output, 'w')
	outputWriter.write('user_id,business_id,percentage' + '\n')
	testReader = open(test, 'r')
	skipLine = testReader.readline()
	for line in testReader:
		line = line.replace('\n', '').split(',')
		user_id = line[0]
		business_id = line[1]
		if user_id not in friends:
			outputWriter.write(user_id + ',' + business_id + ',' + str(0) + '\n')
			continue
		count = 0.0
		for friend in friends[user_id]:
			if (friend, business_id) in reviews:
				count += 1
		percentage = count/len(friends[user_id])
		outputWriter.write(user_id + ',' + business_id + ',' + str(percentage) + '\n')



if __name__ == '__main__':
	args = parse_args()
	# generatePercentage(args)
	
	








