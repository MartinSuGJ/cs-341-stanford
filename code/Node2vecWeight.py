import argparse
from scipy import spatial



def parse_args():

	parser = argparse.ArgumentParser(description="Run node2vec weight.")
	# parser.add_argument('--edge', nargs='?', default='graph/karate.edgelist',
	# 					help='Input graph path')

	parser.add_argument('--fl', nargs='?', default='data/yelp_user_friends_list_train.csv',
						help='Input graph path')	

	parser.add_argument('--emb', nargs='?', default='data/yelp_user_friends_list_train.emb',
						help='Embeddings path')

	parser.add_argument('--weight', nargs='?', default='data/yelp_user_friends_list_train.weight',
						help='Edge weight')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	# print(args)
	# edge = args.edge
	friendList = args.fl
	emb = args.emb
	weight = args.weight

	nodeEmb = {}
	embReader = open(emb, 'r')
	skipLine = embReader.readline()
	for line in embReader:
		line = line.replace('\n','').split(' ')
		nodeEmb[line[0]] = [float(line[i]) for i in range(1, len(line))]

	embReader.close()

	# edgeReader = open(edge, 'r')
	# weightWriter = open(weight, 'w')
	# for line in edgeReader:
	# 	lineList = line.replace("\n","").split(" ")

	# 	score = 1 - spatial.distance.cosine(nodeEmb[lineList[0]], nodeEmb[lineList[1]])
	# 	weightWriter.write(line + ' ' + str(score) + '\n')

	# edgeReader.close()
	# weightWriter.close()

	friendListReader = open(friendList, 'r')
	weightWriter = open(weight, 'w')
	skipLine = friendListReader.readline()
	for line in friendListReader:
		line = line.replace("\n","").split(",")
		userID = line[0]
		numFriend = int(line[1])
		if numFriend == 0:
			weightWriter.write(userID + ',' + str(0) + ',' + '\n')
			continue
		if userID not in nodeEmb:
			weight = [str(1.0/numFriend)]*numFriend
			weightWriter.write(userID + ',' + str(numFriend) + ',' + '::'.join(weight) + '\n')
			continue
		friendList = line[2].split("::")
		weight = []
		scoreSum = 0
		for friend in friendList:
			if friend in nodeEmb:
				score = 1 - spatial.distance.cosine(nodeEmb[userID], nodeEmb[friend])
				scoreSum += score
				weight.append(score)
			else:
				weight.append[0]
		# avg = str(scoreSum/numFriend)
		weight = weight / scoreSum
		weight = [str(w) for w in weight]
		weightWriter.write(userID + ',' + str(numFriend) + ',' + '::'.join(weight) + '\n')
	friendListReader.close()
	weightWriter.close()








