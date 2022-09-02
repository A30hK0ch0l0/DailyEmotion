import pandas as pd
import numpy as np
import tensorflow
from keras import backend as K
import joblib
from parameters import anger_label, wonder_label, fear_label, hate_label, expectation_label, sadness_label
from parameters import happiness_label, credence_label, hope_label, emotion_mean_dic, model_dic
from . import MODULE_ROOT

np.random.seed(4)
tensorflow.random.set_seed(4)


def one_day(data: pd.DataFrame):
    df_list = [0 for i in range(29)]
    df = pd.DataFrame(df_list, index=['time', 'anger', 'wonder', 'fear', 'hate', 'expectation', 'sadness',
                                      'happiness', 'credence', 'hope', 'anger_est', 'anger_label',
                                      'wonder_est', 'wonder_label', 'fear_est', 'fear_label', 'hate_est',
                                      'hate_label', 'expectation_est', 'expectation_label', 'sadness_est',
                                      'sadness_label', 'happiness_est', 'happiness_label', 'credence_est',
                                      'credence_label', 'hope_est', 'hope_label', 'gaussian']).T

    df["time"].iloc[0] = data["time"][7]
    df["anger"].iloc[0] = data["anger"][7]
    df["wonder"].iloc[0] = data["wonder"][7]
    df["fear"].iloc[0] = data["fear"][7]
    df["hate"].iloc[0] = data["hate"][7]
    df["expectation"].iloc[0] = data["expectation"][7]
    df["sadness"].iloc[0] = data["sadness"][7]
    df["happiness"].iloc[0] = data["happiness"][7]
    df["credence"].iloc[0] = data["credence"][7]
    df["hope"].iloc[0] = data["hope"][7]

    est_emotion = ["anger_est", "wonder_est", "fear_est", "hate_est", "expectation_est", "sadness_est", "happiness_est", "credence_est", "hope_est"]

    for em in est_emotion:
        data_normalised = np.array(data[em[:-4]]) / emotion_mean_dic[em[:-4]]
        ohlcv_histories_normalised = np.array(data_normalised[: -1]).reshape(1, 7)

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

        model = tensorflow.keras.models.load_model(model_dic[em[:-4]])
        y_test_predicted = model.predict([x_test1, tech_ind_test])
        y_test_predicted = y_test_predicted * emotion_mean_dic[em[:-4]]
        df[em].iloc[0] = y_test_predicted[-1][0]

        em_label = em[:-4] + "_label"
        if em_label == "anger_label":
            df[em_label].iloc[0] = anger_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "wonder_label":
            df[em_label].iloc[0] = wonder_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "fear_label":
            df[em_label].iloc[0] = fear_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "hate_label":
            df[em_label].iloc[0] = hate_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "expectation_label":
            df[em_label].iloc[0] = expectation_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "sadness_label":
            df[em_label].iloc[0] = sadness_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "happiness_label":
            df[em_label].iloc[0] = happiness_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "credence_label":
            df[em_label].iloc[0] = credence_label(df[em[:-4]][0] - df[em][0])
        elif em_label == "hope_label":
            df[em_label].iloc[0] = hope_label(df[em[:-4]][0] - df[em][0])

    scaler = joblib.load(f"{MODULE_ROOT}/Data/10_scaler.sav")
    gaussian = joblib.load(f"{MODULE_ROOT}/Data/11_gaussian.sav")

    tabel = df[['anger', 'wonder', 'fear', 'hate', 'expectation', 'sadness', 'happiness', 'credence', 'hope',
                "anger_label", "wonder_label", "fear_label", "hate_label", "expectation_label",
                "sadness_label", "happiness_label", "credence_label", "hope_label"]]

    df_normalized = scaler.transform(tabel)
    gaussian_predict = gaussian.predict(df_normalized)
    df["gaussian"].iloc[0] = gaussian_predict
    return df
