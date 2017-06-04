from MatrixFactorizationFriendsModified import MatrixFactorization
import os
import argparse
import time

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outputDirName", help="name of the output file")
    parser.add_argument("predictType", help="type of the prediction. 1:predict all; 2:improve coverage rate; 3:ignore 234", type=int)
    parser.add_argument("tList", help="parameter of t")
    parser.add_argument("numIterList", help="parameter of number of iteration")
    parser.add_argument("etaList", help="parameter of step size")
    parser.add_argument("numLFList", help="parameter of dimension of latent factor")
    parser.add_argument("lambdaList", help="parameter of regularization term")

    args = parser.parse_args()
    outputDirName = args.outputDirName
    predictType = args.predictType
    tList = [float(i) for i in args.tList.split("_")]
    numIterList = [int(i) for i in args.numIterList.split("_")]
    etaList = [float(i) for i in args.etaList.split("_")]
    numLFList = [int(i) for i in args.numLFList.split("_")]
    lambdaList = [float(i) for i in args.lambdaList.split("_")]

    parentDir = "/home/guangjun/"
    # parentDir = "/Users/guangjun/Desktop/yelp_data_challenge/"
    outputFile = parentDir + "result/" + outputDirName
    if not os.path.exists(outputFile):
        os.makedirs(outputFile)
    # socialFileTrain = parentDir + "data/yelp_user_friends_list_train.csv"
    # socialFileTrainAndValidate = parentDir + "data/yelp_user_friends_list_trainAndValidate.csv"
    socialFileTrain = parentDir + "data/yelp_user_friends_list_train_top.csv"
    socialFileTrainAndValidate = parentDir + "data/yelp_user_friends_list_trainAndValidate_top.csv"
    trainFile = parentDir + "data/t_train.csv"
    validateFile = parentDir + "data/t_validate.csv"
    trainAndValidate = parentDir + "data/t_trainAndValidate.csv"
    testFile = parentDir + "data/t_test.csv"

    with open(outputFile+"/final_result_"+time.strftime("%Y%m%d_%H%M%S")+".txt", "w") as resultFile:
        # Grid search step
        print("====================Grid Search Step====================")
        mf1 = MatrixFactorization(socialFileTrain, outputFile,trainFile, validateFile, 720165, 48059, predictType)
        # numIterList = [20]
        # etaList = [10e-2,10e-3]
        # numLFList = [4,6,8]
        # lambdaList = [0.3]
        bestModel = mf1.gridSeacrh(tList = tList, numIterList = numIterList, etaList = etaList, numLFList = numLFList, lambdaList = lambdaList)
        bestModelString = ("Best model is eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n" % (bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter, bestModel.t))
        print(bestModelString)
        resultFile.write(bestModelString)

        # compute the remse error of the test set
        print("====================Test Step====================")

        mf2 = MatrixFactorization(socialFileTrainAndValidate, outputFile,trainAndValidate, testFile, 720165, 48059, predictType)
        (rmseAll1,rmseAll2,rmseImprove,rmsePart) = mf2.test(bestModel)
        rmseAll1String = ("RMSE(All1) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseAll1, bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter, bestModel.t))
        rmseAll2String = ("RMSE(All2) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseAll2, bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter, bestModel.t))
        rmseImproveString = ("RMSE(Improve) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmseImprove, bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter, bestModel.t))
        rmsePartString = ("RMSE(Part) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d, t = %f\n"  % (rmsePart, bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter, bestModel.t))
        resultFile.write(rmseAll1String)
        resultFile.write(rmseAll2String)
        resultFile.write(rmseImproveString)
        resultFile.write(rmsePartString)
