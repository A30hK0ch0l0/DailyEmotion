import pandas as pd
import datetime
import numpy as np
import tensorflow
from keras import backend as K
# from parameters import emotion_mean_dic, end_time, model_dic
from daily_emotions.parameters import emotion_mean_dic, model_dic

np.random.seed(4)
tensorflow.random.set_seed(4)


def days3(data: pd.DataFrame):
    data_time = (data['time'][len(data)-1])
    end_time = (datetime.datetime.strptime(data_time, '%Y-%m-%d')).date()
    time = []
    for i in range(3):
        start_time1 = end_time + datetime.timedelta(days=i + 1)
        # start_time1 = (datetime.datetime(int(end_time[:4]), int(end_time[5:7]), int(end_time[8:])) + datetime.timedelta(days=i + 1)).date()
        start_time = f"{start_time1.year}-{start_time1.month}-{start_time1.day}"
        time.append(start_time)

    df = pd.DataFrame(time, columns=["time"])

    est_emotion = ["anger_est", "wonder_est", "fear_est", "hate_est", "expectation_est", "sadness_est", "happiness_est", "credence_est", "hope_est"]

    for em in est_emotion:
        data_list = data[em[:-4]].tolist()
        for i in range(3):
            data_normalised = np.array(data_list) / emotion_mean_dic[em[:-4]]
            # history_points = 7

            ohlcv_histories_normalised = np.array(data_normalised[-7:]).reshape(1, 7)

            technical_indicators = []
            for his in ohlcv_histories_normalised:
                # note since we are using his[3] we are taking the SMA of the closing price
                sma = np.mean(his)
                technical_indicators.append(np.array([sma, ]))

            technical_indicators = np.array(technical_indicators)

            ohlcv_test = ohlcv_histories_normalised
            tech_ind_test = technical_indicators

            x_test1 = K.cast_to_floatx(ohlcv_test)
            tech_ind_test = K.cast_to_floatx(tech_ind_test)
            # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

            model = tensorflow.keras.models.load_model(model_dic[em[:-4]])
            y_test_predicted = model.predict([x_test1, tech_ind_test])
            y_test_predicted = y_test_predicted * emotion_mean_dic[em[:-4]]
            data_list.append(y_test_predicted[0][0])
        df[em] = data_list[-3:]
    return df


