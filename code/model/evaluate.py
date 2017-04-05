# -*- coding: utf-8 -*-

# Evaluation
# Function to evaluate the performance of the recommender system
# RMSE
import pandas as pd

def evaluate(truth_file_path, predict_file_path, metric):
    """
    truth_file_path:     csv file
    predict_file_path:   csv file
    metric:              'RMSE'
    """
    truth_file = pd.read_csv(truth_file_path)
    predict_file = pd.read_csv(predict_file_path)

    all_file = pd.merge(truth_file, predict_file, how='left', on=['user_id', 'business_id'])
    if metric == 'RMSE':
        rmse = np.sqrt(np.mean(sum((all_file[truth_file.columns[2]+'_x'] - all_file[predict_file.columns[2]+'_y'])**2)))
        return rmse
    else:
        return 0
