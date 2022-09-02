import pandas as pd
import tensorflow
import numpy as np
from keras import backend as K
import joblib
from .parameters import anger_label, wonder_label, fear_label, hate_label, expectation_label, sadness_label
from .parameters import happiness_label, credence_label, hope_label, emotion_mean_dic, model_dic
from . import MODULE_ROOT

np.random.seed(4)
tensorflow.random.set_seed(4)


def all_time(data: pd.DataFrame) -> pd.DataFrame:
    df = data[7:]

    est_emotion = ["anger_est", "wonder_est", "fear_est", "hate_est", "expectation_est", "sadness_est", "happiness_est", "credence_est", "hope_est"]

    for em in est_emotion:
        data_normalised = np.array(data[em[:-4]]) / emotion_mean_dic[em[:-4]]
        history_points = 7

        ohlcv_histories_normalised = np.array([data_normalised[i: i + history_points] for i in range(len(data_normalised) - history_points)])

        technical_indicators = []
        for his in ohlcv_histories_normalised:
            # note since we are using his[3] we are taking the SMA of the closing price
            sma = np.mean(his)
            technical_indicators.append(np.array([sma, ]))

        technical_indicators = np.array(technical_indicators)

        ohlcv_test = ohlcv_histories_normalised
        tech_ind_test = technical_indicators
        # y_test = next_day_anger_normalised
        # unscaled_y_test = next_day_anger_values

        x_test1 = K.cast_to_floatx(ohlcv_test)
        tech_ind_test = K.cast_to_floatx(tech_ind_test)
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

        model = tensorflow.keras.models.load_model(model_dic[em[:-4]])
        y_test_predicted = model.predict([x_test1, tech_ind_test])
        y_test_predicted = y_test_predicted * emotion_mean_dic[em[:-4]]
        df[em] = y_test_predicted

        em_label = em[:-4] + "_label"
        if em_label == "anger_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: anger_label(x))
        elif em_label == "wonder_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: wonder_label(x))
        elif em_label == "fear_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: fear_label(x))
        elif em_label == "hate_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: hate_label(x))
        elif em_label == "expectation_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: expectation_label(x))
        elif em_label == "sadness_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: sadness_label(x))
        elif em_label == "happiness_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: happiness_label(x))
        elif em_label == "credence_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: credence_label(x))
        elif em_label == "hope_label":
            df[em_label] = (df[em[:-4]] - df[em]).apply(lambda x: hope_label(x))

    df.reset_index(inplace=True, drop=True)

    scaler = joblib.load(f"{MODULE_ROOT}/Data/10_scaler.sav")
    gaussian = joblib.load(f"{MODULE_ROOT}/Data/11_gaussian.sav")

    tabel = df[['anger', 'wonder', 'fear', 'hate', 'expectation', 'sadness', 'happiness', 'credence', 'hope',
                "anger_label", "wonder_label", "fear_label", "hate_label", "expectation_label",
                "sadness_label", "happiness_label", "credence_label", "hope_label"]]

    df_normalized = scaler.transform(tabel)
    df["gaussian"] = gaussian.predict(df_normalized)
    return df
