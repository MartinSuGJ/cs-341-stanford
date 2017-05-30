# import pandas as pd



# friendRating = pd.read_csv('nonthreshold_runFriendRating_local_predict_1_20170527_044317.csv')
# friend = pd.read_csv('nonthreshold_runFriend1_low_3_predict_1_20170529_051503.csv')
# lf = pd.read_csv('nonthreshold_run1_low_5_predict_1_20170527_095857.csv')



# def parse_args():
# 	parser = argparse.ArgumentParser(description='verify friend function')
# 	parser.add_argument('--input', nargs='?', default=)


import pandas as pd
friendPercent = pd.read_csv('test_reviewPecent.csv')
friendRate = friendPercent[friendPercent.percentage > 0]
friendRate['rate_friend_num'] = friendRate['friend_num'] * friendRate['percentage']
