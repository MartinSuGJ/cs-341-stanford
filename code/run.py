import time
import os
import argparse
from MatrixFactorization import MatrixFactorization



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outputDirName", help="name of the output file")
    parser.add_argument("predictType", help="type of the prediction. 1:predict all; 2:improve coverage rate; 3:ignore 234", type=int)
    parser.add_argument("numIterList", help="parameter of number of iteration")
    parser.add_argument("etaList", help="parameter of step size")
    parser.add_argument("numLFList", help="parameter of dimension of latent factor")
    parser.add_argument("lambdaList", help="parameter of regularization term")

    args = parser.parse_args()
    outputDirName = args.outputDirName
    predictType = args.predictType
    numIterList = [int(i) for i in args.numIterList.split("_")]
    etaList = [float(i) for i in args.etaList.split("_")]
    numLFList = [int(i) for i in args.numLFList.split("_")]
    lambdaList = [float(i) for i in args.lambdaList.split("_")]

    parentDir = "/home/guangjun/"
    # parentDir = "/Users/guangjun/Desktop/yelp_data_challenge/"
    outputFile = parentDir + "result/" + outputDirName
    if not os.path.exists(outputFile):
        os.makedirs(outputFile)
    trainFile = parentDir + "data/train.csv"
    validateFile = parentDir + "data/validate.csv"
    trainAndValidate = parentDir + "data/trainAndValidate.csv"
    testFile = parentDir + "data/test.csv"

    with open(outputFile+"/final_result_"+time.strftime("%Y%m%d_%H%M%S")+".txt", "w") as resultFile:
        # Grid search step
        print("====================Grid Search Step====================")
        mf1 = MatrixFactorization(outputFile,trainFile, validateFile, 720165, 48059, predictType)
        bestModel = mf1.gridSeacrh(numIterList = numIterList, etaList = etaList, numLFList = numLFList, lambdaList = lambdaList)
        bestModelString = ("Best model is eta = %f, rank = %d, lambda = %f, num of iteration = %d\n" % (bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter))
        print(bestModelString)
        resultFile.write(bestModelString)

        # compute the remse error of the test set
        print("====================Test Step====================")

        mf2 = MatrixFactorization(outputFile,trainAndValidate, testFile, 720165, 48059, predictType)
        (rmseAll1,rmseAll2,rmsePart) = mf2.test(bestModel)
        rmseAll1String = ("RMSE(All1) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d\n"  % (rmseAll1, bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter))
        rmseAll2String = ("RMSE(All2) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d\n"  % (rmseAll2, bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter))
        rmsePartString = ("RMSE(Part) = %f for the model trained with eta = %f, rank = %d, lambda = %f, num of iteration = %d\n"  % (rmsePart, bestModel.bestEta, bestModel.bestNumLF, bestModel.bestLambda, bestModel.bestNumIter))
        resultFile.write(rmseAll1String)
        resultFile.write(rmseAll2String)
        resultFile.write(rmsePartString)
