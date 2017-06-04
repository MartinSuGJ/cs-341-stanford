# import pandas as pd
import argparse
import itertools

# friendRating = pd.read_csv('nonthreshold_runFriendRating_local_predict_1_20170527_044317.csv')
# friend = pd.read_csv('nonthreshold_runFriend1_low_3_predict_1_20170529_051503.csv')
# lf = pd.read_csv('nonthreshold_run1_low_5_predict_1_20170527_095857.csv')



def parse_args():
	parser = argparse.ArgumentParser(description='verify friend variance < all variance')
	parser.add_argument('--input', nargs='?', default='data/test.csv')
	parser.add_argument('--friend', nargs='?', default='data/yelp_user_friends_list_trainAndValidate.csv')


if __name__ == '__main__':
	args = parse_args()
	inputFile = args.input
	friend = args.friend
	
	rateList = {}
	inputReader = open(inputFile, 'r')
	skipLine = inputReader.readline()
	for line in inputReader:
		line = line.replace('\n', '').split(',')
		if line[1] not in rateList:
			rateList[line[1]] = {}
		rateList[line[1]][line[0]] = float(line[2])
	inputReader.close()


	friendList = {}
	friendReader = open(friend, 'r')
	skipLine = friendReader.readline()
	for line in friendReader:
		line = line.replace('\n', '').split(',')
		friendList[line[0]] = line[2].split('::')
	friendReader.close()


	allSum = 0
	allCount = 0
	friendSum = 0
	friendCount = 0
	for business in rateList.keys():
		pairs = list(itertools.combinations(rateList[business]))
		
		var = (rateList[business][pairs[0]] - rateList[business][pairs[1]])**2
		allCount += 1
		allSum += var
		if (pairs[0] in friendList[pairs[1]]) or (pairs[1] in friendList[pairs[0]])
			friendSum += var
			friendCount += 1

	print(friendSum*allCount/(friendCount*allSum))


	




# import pandas as pd
# import numpy as np
# friendPercent = pd.read_csv('test_reviewPecent.csv')
# friendRate = friendPercent[friendPercent.percentage > 0]
# friendRate['rate_friend_num'] = friendRate['friend_num'] * friendRate['percentage']
# friendRate['diff'] = np.abs(friendRate['friend_rate'] - friendRate['rate'])
# outlier = friendRate[friendRate['diff']>=3]


# If there is only one friend that rated business. Then 
