# import pandas as pd
# from estimate_emotion.one_day_estimate import one_day
# from estimate_emotion.days3_prediction import days3
# from estimate_emotion.all_time_estimate import all_time
#
# data = pd.read_pickle("./Data/One_day_Emotion.pickle")
# ans_one_day = one_day(data)
# ans_days3 = days3(data)
# print("ans_one_day: ", ans_one_day)
# print("\n ----------- \n")
# print("\n ans_days3: ", ans_days3)
import os

MODULE_ROOT = f'{os.path.dirname(os.path.abspath(__file__))}'
