	

import argparse
import pandas as pd
import numpy as np

def parse_args():

	parser = argparse.ArgumentParser(description='Calculate valid friend review percentage')
	# --type: generate/analysis
	parser.add_argument('--type', nargs='?', default='generate')
	parser.add_argument('--trainAndVal', nargs='?', default='data/trainAndValidate.csv')
	parser.add_argument('--test', nargs='?', default='data/test.csv')
	parser.add_argument('--friendList', nargs='?', default='data/yelp_user_friends_list_trainAndValidate_top.csv')
	parser.add_argument('--output', nargs='?', default='data/test_reviewPecent_top.csv')
	parser.add_argument('--Ainput', nargs='?', default='data/test_reviewPecent.csv')

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
			reviews[(line[0], line[1])] = float(line[2])
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
	outputWriter.write('user_id,business_id,friend_num,percentage,friend_rate,rate,diff,friend_rate_num' + '\n')
	testReader = open(test, 'r')
	skipLine = testReader.readline()
	for line in testReader:
		line = line.replace('\n', '').split(',')
		user_id = line[0]
		business_id = line[1]
		rate = line[2]
		if user_id not in friends:

			outputWriter.write(','.join([user_id,business_id,str(0),str(0),str(0),rate,rate,str(0)]) +'\n')
			continue
		count = 0.0
		friendSum = 0.0
		for friend in friends[user_id]:
			if (friend, business_id) in reviews:
				count += 1
				friendSum += reviews[(friend, business_id)]
		percentage = count/len(friends[user_id])
		if count > 0:
			outputWriter.write(','.join([user_id,business_id,str(len(friends[user_id])),str(percentage),str(friendSum/count),rate,str(np.abs(float(rate)-friendSum/count)),str(count)])+'\n')
		else:
			outputWriter.write(','.join([user_id,business_id,str(len(friends[user_id])),str(percentage),str(0),rate,rate,str(0)])+'\n')





def analysis(args):
	Ainput = args.Ainput

	percentage = pd.read_csv(Ainput)

	# percentage = {}
	# inputReader = open(Ainput, 'r')
	# skipLine = inputReader.readline()
	# for line in inputReader:
	# 	line = line.replace('\n', '').split(',')
	# 	percentage[(line[0], line[1])] = float(line[2])





if __name__ == '__main__':
	args = parse_args()
	if args.type == 'generate':
		generatePercentage(args)
	elif args.type == 'analysis':
		analysis(args)


	








