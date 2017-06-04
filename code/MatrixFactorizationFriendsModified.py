"""
Stochastic Gradient Descent
- Input
    userID, businessID, stars, globalAvg
    userNum: total number of users, say m
    bizNum: total number of business, say n
    eta: step size
    numLF: number of latend factors
    lambdaBiz: regularization parameter of business biases term
    lambdaUser: regularization parameter of user biases term
    lambdaLF1: regularization parameter of user's latent factor
    lambdaLF2: regularization parameter of business's latent factor
    numIter: number of iteration
    tol:
- Output
    businessBiases: n*1 vector
    userBiases: m*1 vector
    latentFactorUser: k*m matrix
    latentFactorBusiness: k*n matrix
"""

import numpy as np
import random as rd
import itertools
import math
import time

userNum = 720165
bizNum = 48059

"""
Model class
"""
class BestModel(object):
    def __init__(self,t,bestNumIter,bestTol,bestEta,bestNumLF,bestLambda,P,Q,B,U):
        self.t = t
        self.bestNumIter = bestNumIter
        self.bestTol = bestTol
        self.bestEta = bestEta
        self.bestNumLF = bestNumLF
        self.bestLambda = bestLambda
        self.P = P
        self.Q = Q
        self.B = B
        self.U = U

"""
Initialize the matrix P and Q
"""

class MatrixFactorization(object):
    def __init__(self,
                 socialFile,
                 outputFile,
                 trainFile,
                 testFile,
                 userNum,
                 bizNum,
                 predictType,
                 t = 0.8,
                 numIter = 20,
                 tol = 0.1,
                 eta = 10e-5,
                 numLF =4,
                 lambdaUser = 0.0,
                 lambdaBiz = 0.0,
                 lambdaLF1 = 0.0,
                 lambdaLF2 = 0.0):
        """
        Train a matrix factrization model to predict empty
        entries in a matrix. We implement Stochastic
        Gradient Descent method to solve the optimization
        problem.

        Parameters
        ==========
        trainFile: name of the training dataset
        testFile: name of the testing dataset
        userNum: number of the users in total
        bizNum: number of business in total
        numIter: number of iteration to scan the file
        tol: tolerance of the convergence
        eta: step size
        numLF: dimension of the latent factors
        lambdaUser: regularization term for user biases
        lambdaBiz: regularization term for business biases
        lambdaLF1: regularization term for user latent factors
        lambdaLF2: regularization term for business latent factors
        """
        self.socialFile = socialFile
        self.outputFile = outputFile
        self.trainFile = trainFile
        self.testFile = testFile
        self.userNum = userNum
        self.bizNum = bizNum
        self.predictType = predictType
        self.numIter = numIter
        self.tol = tol
        self.eta = eta
        self.numLF = numLF
        self.t = t
        self.lambdaUser = lambdaUser
        self.lambdaBiz = lambdaBiz
        self.lambdaLF1 = lambdaLF1
        self.lambdaLF2 = lambdaLF2
        # Read the social network file
        self.social = dict()
        with open(socialFile, "r") as social:
            skipLine = social.readline()
            for line in social:
                line = line.replace("\n","").split(",")
                userID = int(line[0])
                numFriend = int(line[1])
                friendList = line[2].split("::")
                if numFriend == 0:
                    self.social[userID] = []
                else:
                    friendList = [int(i) for i in friendList]
                    self.social[userID] = friendList

        # Read the train file and test file to the dictionary
        self.trainData = dict()
        with open(self.trainFile, "r") as trainF:
            skipLine = trainF.readline()
            counter = 0
            for line in trainF:
                line = line.replace("\n","").split(",")
                userID = int(line[0])
                bizID = int(line[1])
                stars = float(line[2])
                globalAvg = float(line[3])
                userAvg = float(line[4])
                bizAvg = float(line[5])
                state = line[6]
                self.trainData[counter] = {"userID":userID, "bizID":bizID, "stars":stars, "globalAvg":globalAvg, "userAvg":userAvg, "bizAvg":bizAvg, "state":state}
                counter += 1
        self.trainNum = counter

        self.testData = dict()
        with open(self.testFile, "r") as testF:
            skipLine = testF.readline()
            counter = 0
            for line in testF:
                line = line.replace("\n","").split(",")
                userID = int(line[0])
                bizID = int(line[1])
                stars = float(line[2])
                globalAvg = float(line[3])
                userAvg = float(line[4])
                bizAvg = float(line[5])
                state = line[6]
                group = int(line[7])
                self.testData[counter] = {"userID":userID, "bizID":bizID, "stars":stars, "globalAvg":globalAvg, "userAvg":userAvg, "bizAvg":bizAvg, "state":state, "group":group}
                counter += 1
        self.testNum = counter

    def train(self):
        resultFilename = "train_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
        resultFile = open(self.outputFile + '/' + resultFilename, "w")
        """
        Initialize the matrix P and Q
            - each entries are random values within [0,sqrt(5/k)]
        """
        self.P = math.sqrt(5/float(self.numLF))*np.random.rand(self.userNum,self.numLF)
        self.Q = math.sqrt(5/float(self.numLF))*np.random.rand(self.bizNum,self.numLF)

        """
        Initialize the matrix F (Friends Latent Factor Matrix)
            - each entries are random values within [0,sqrt(5/k)]
        """
        self.F = math.sqrt(5/float(self.numLF))*np.random.rand(self.userNum,self.numLF)

        """
        Initialize the user biases and business biases
        """
        self.B = np.random.rand(self.bizNum) - 0.5
        self.U = np.random.rand(self.userNum) - 0.5

        """
        Iterate and update the value
        """
        error = []
        resultFile.write("Model:eta = %f, rank = %d, lambda = %f, num of iteration = %d =====>\n" % (float(self.eta), int(self.numLF), float(self.lambdaBiz), int(self.numIter)))
        for iteration in xrange(self.numIter):
            print("iteration: %d" % iteration)
            indexArray = range(self.trainNum)
            rd.shuffle(indexArray)
            for index in indexArray:
                record = self.trainData[index]
                u = record["userID"]
                i = record["bizID"]
                r = record["stars"]
                mu = record["globalAvg"]
                friendList = self.social[u]
                numFriend = len(friendList)

                bi = self.B[i]
                uu = self.U[u]
                qi = self.Q[i,]
                pu = self.P[u,]

                # complicate term about friends P vectors
                pfu = 0
                for friend in friendList:
                    pfu += self.F[friend,]
                pf = self.t*pu + (1-self.t)/(numFriend+1)*(pu + pfu)

                e = r - (mu + bi + uu + np.dot(qi, pf))

                bizBiasesNew = bi + self.eta*(e - self.lambdaBiz*bi)
                userBiasesNew = uu + self.eta*(e - self.lambdaUser*uu)
                latentFactorUserNew = pu + self.eta*(e*(self.t*qi + (1-self.t)/float(numFriend + 1)*qi) - self.lambdaLF1*pu)
                latentFactorBizNew = qi + self.eta*(e*pf - self.lambdaLF2*qi)

                # Update the value
                self.B[i] = bizBiasesNew
                self.U[u] = userBiasesNew
                self.P[u,] = latentFactorUserNew
                self.Q[i,] = latentFactorBizNew
                for friend in friendList:
                    self.F[friend,] = self.F[friend,] + self.eta*(e*(1-self.t)/float(numFriend+1)*qi - self.lambdaLF1*self.F[friend,])

            err = 0
            for index in xrange(self.trainNum):
                record = self.trainData[index]
                u = record["userID"]
                i = record["bizID"]
                r = record["stars"]
                mu = record["globalAvg"]
                friendList = self.social[u]
                numFriend = len(friendList)


                bi = self.B[i]
                uu = self.U[u]
                qi = self.Q[i,]
                pu = self.P[u,]

                # complicate term about friends P vectors
                pfu = 0
                for friend in friendList:
                    pfu += self.F[friend,]
                pf = self.t*pu + (1-self.t)/(numFriend+1)*(pu + pfu)

                err += (r - (mu + bi + uu + np.dot(qi,pf)))**2

            # regularization for user biases term
            err += self.lambdaUser*np.dot(self.U, self.U)
            # regularization for business biases term
            err += self.lambdaBiz*np.dot(self.B, self.B)
            # regularization for latent factors term
            for user in xrange(self.userNum):
                err += self.lambdaLF1*np.dot(self.P[user,],self.P[user,])
                err += self.lambdaLF1*np.dot(self.F[user,],self.F[user,])
                # err += self.lambdaUser*self.U[user]*self.U[user]
            for biz in xrange(self.bizNum):
                err += self.lambdaLF2*np.dot(self.Q[biz,],self.Q[biz,])
                # err += self.lambdaBiz*self.B[biz]*self.B[biz]
            error.append(float(err))
            resultFile.write("Training step: error is %f at iteration %d\n" % (err, iteration))
            print("Training step: error is %f at iteration %d\n" % (err, iteration))
        resultFile.close()
        return None

    def predict(self, predictType, isOutput=False):
        squareError = 0
        counter = 0
        if isOutput:
            predictFilename = "predict_" + str(predictType) + "_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
            predictFile = open(self.outputFile + '/' + predictFilename, "w")
            predictFile.write("userID,bizID,stars,group,numFriend,globalAvg,userAvg,bizAvg,predictScore\n")

        for index in xrange(self.testNum):
            record = self.testData[index]
            u = record["userID"]
            i = record["bizID"]
            r = record["stars"]
            mu = record["globalAvg"]
            userAvg = record["userAvg"]
            bizAvg = record["bizAvg"]
            group = record["group"]
            friendList = self.social[u]
            numFriend = len(friendList)

            bi = self.B[i]
            uu = self.U[u]
            qi = self.Q[i,]
            pu = self.P[u,]

            # complicate term about friends P vectors
            pfu = 0
            for friend in friendList:
                pfu += self.F[friend,]
            pf = self.t*pu + (1-self.t)/(numFriend+1)*(pu + pfu)

            predictScore = max(min((mu + bi + uu + np.dot(qi,pf)),5),0)

            if isOutput:
                predictFile.write("%d,%d,%f,%d,%d,%f,%f,%f,%f\n" % (u,i,r,group,numFriend,mu,userAvg,bizAvg,predictScore))

            if predictType == 1:
                # predict all
                counter += 1
                if group == 1:
                    squareError += (r - predictScore)**2
                elif group == 2:
                    squareError += (r - userAvg)**2
                elif group == 3:
                    squareError += (r - bizAvg)**2
                else:
                    squareError += (r - mu)**2
            elif predictType == 2:
                # improve the coverage rate
                if group == 1:
                    counter += 1
                    squareError += (r - predictScore)**2
                elif group == 3:
                    counter += 1
                    if numFriend > 0:
                        squareError += (r - (mu + bi + 0 + np.dot(qi,1/float(numFriend)*pfu)))**2
                    else:
                        squareError += (r - bizAvg)**2
                else:
                    squareError += 0
            elif predictType == 3:
                # only predict the group 1 record
                if group == 1:
                    counter += 1
                    squareError += (r - (mu + bi + uu + np.dot(qi,pf)))**2
                else:
                    squareError += 0
            else:
                counter += 1
                squareError += (r - (mu + bi + uu + np.dot(qi,pf)))**2

        if isOutput:
            predictFile.close()

        rmse = np.sqrt(squareError/float(counter))
        return rmse

    def gridSeacrh(self, tList=[0.8], numIterList=[20], tolList=[0.1], etaList=[10e-5], numLFList=[4], lambdaList=[0.0]):
        resultFilename = "grid_search_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
        resultFile = open(self.outputFile + "/" + resultFilename, "w")
        bestRMSE = float("inf")
        bestT = -1
        bestNumIter = -1
        bestTol = -1
        bestEta = -1
        bestNumLF = -1
        bestLambda = -1
        for t, numIter, tol, eta, numLF, lmbda in itertools.product(tList, numIterList, tolList, etaList, numLFList, lambdaList):
            self.t = t
            self.numIter = numIter
            self.tol = tol
            self.eta = eta
            self.numLF = numLF
            self.lambdaBiz = lmbda
            self.lambdaUser = lmbda
            self.lambdaLF1 = lmbda
            self.lambdaLF2 = lmbda

            # Train step
            self.train()
            # Test step
            rmse = self.predict(self.predictType)
            if rmse < bestRMSE:
                bestRMSE = rmse
                bestT = self.t
                bestNumIter = self.numIter
                bestTol = self.tol
                bestEta = self.eta
                bestNumLF = self.numLF
                bestLambda = lmbda
                bestP = self.P
                bestQ = self.Q
                bestB = self.B
                bestU = self.U

            resultFile.write("RMSE = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n" % (rmse, eta, numLF, lmbda, numIter, t))


        resultFile.write("Best model is eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n" % (bestEta, bestNumLF, bestLambda, bestNumIter, bestT))
        resultFile.write("=========================================================================\n")
        resultFile.close()
        bestModel = BestModel(bestT, bestNumIter, bestTol, bestEta, bestNumLF, bestLambda, bestP, bestQ, bestB, bestU)
        return bestModel

    def test(self, bestModel):
        self.t = bestModel.t
        self.numIter = bestModel.bestNumIter
        self.tol = bestModel.bestTol
        self.eta = bestModel.bestEta
        self.numLF = bestModel.bestNumLF
        self.lambdaBiz = bestModel.bestLambda
        self.lambdaUser = bestModel.bestLambda
        self.lambdaLF1 = bestModel.bestLambda
        self.lambdaLF2 = bestModel.bestLambda

        # train step
        self.train()
        # test step
        rmseAll1 = self.predict(1, True)
        rmseImprove = self.predict(2, True)
        rmsePart = self.predict(3, True)
        rmseAll2 = self.predict(4, True)

        print("RMSE(All1) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseAll1, self.eta, self.numLF, self.lambdaBiz, self.numIter, self.t))
        print("RMSE(All2) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseAll2, self.eta, self.numLF, self.lambdaBiz, self.numIter, self.t))
        print("RMSE(Improve) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseImprove, self.eta, self.numLF, self.lambdaBiz, self.numIter, self.t))
        print("RMSE(Part) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmsePart, self.eta, self.numLF, self.lambdaBiz, self.numIter, self.t))

        return rmseAll1,rmseAll2,rmseImprove,rmsePart

    def getRMSE(self, bestModel):
        self.P = bestModel.P
        self.Q = bestModel.Q
        self.B = bestModel.B
        self.U = bestModel.U
        eta = bestModel.bestEta
        numLF = bestModel.bestNumLF
        lmbda = bestModel.bestLambda
        numIter = bestModel.bestNumIter
        t = bestModel.t

        rmseAll1 = self.predict(1)
        rmseImprove = self.predict(2)
        rmsePart = self.predict(3)
        rmseAll2 = self.predict(4)

        print("RMSE(All1) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseAll1, eta, numLF, lmbda, numIter, t))
        print("RMSE(All2) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseAll2, eta, numLF, lmbda, numIter, t))
        print("RMSE(Improve) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseImprove, eta, numLF, lmbda, numIter, t))
        print("RMSE(Part) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmsePart, eta, numLF, lmbda, numIter, t))

        return rmseAll1,rmseAll2,rmseImprove,rmsePart
