from daily_emotions import MODULE_ROOT

emotion_mean_dic = {
    "anger": 0.03633226611545325,
    "wonder": 0.05646248898231598,
    "fear": 0.014688971323794618,
    "hate": 0.02058321646666575,
    "expectation": 0.06015174899494842,
    "sadness": 0.017620016082788643,
    "happiness": 0.055235159626954175,
    "credence": 0.051266797795185906,
    "hope": 0.036382221699082125
}

model_dic = {
    "anger": f"{MODULE_ROOT}/Data/01_anger.h5",
    "wonder": f"{MODULE_ROOT}/Data/02_wonder.h5",
    "fear": f"{MODULE_ROOT}/Data/03_fear.h5",
    "hate": f"{MODULE_ROOT}/Data/04_hate.h5",
    "expectation": f"{MODULE_ROOT}/Data/05_expectation.h5",
    "sadness": f"{MODULE_ROOT}/Data/06_sadness.h5",
    "happiness": f"{MODULE_ROOT}/Data/07_happiness.h5",
    "credence": f"{MODULE_ROOT}/Data/08_credence.h5",
    "hope": f"{MODULE_ROOT}/Data/09_hope.h5"
}


def anger_label(x):
    if x >= 0.00577668:
        return 4
    elif 0.00181377 <= x < 0.00577668:
        return 3
    elif -0.00034719 <= x < 0.00181377:
        return 2
    elif -0.0031419 <= x < -0.00034719:
        return 1
    else:
        return 0


def wonder_label(x):
    if x >= 0.00533986:
        return 4
    elif 0.0017327 <= x < 0.00533986:
        return 3
    elif -0.00110033 <= x < 0.0017327:
        return 2
    elif -0.00535489 <= x < -0.00110033:
        return 1
    else:
        return 0


def fear_label(x):
    if x >= 0.00181935:
        return 4
    elif 0.000466155 <= x < 0.00181935:
        return 3
    elif -0.000478504 <= x < 0.000466155:
        return 2
    elif -0.00159112 <= x < -0.000478504:
        return 1
    else:
        return 0


def hate_label(x):
    if x >= 0.00577134:
        return 4
    elif 0.00147908 <= x < 0.00577134:
        return 3
    elif -0.000336974 <= x < 0.00147908:
        return 2
    elif -0.00254102 <= x < -0.000336974:
        return 1
    else:
        return 0


def expectation_label(x):
    if x >= 0.00902604:
        return 4
    elif 0.00130229 <= x < 0.00902604:
        return 3
    elif -0.00224967 <= x < 0.00130229:
        return 2
    elif -0.00690723 <= x < -0.00224967:
        return 1
    else:
        return 0


def sadness_label(x):
    if x >= 0.00286893:
        return 4
    elif 0.0010953 <= x < 0.00286893:
        return 3
    elif -9.53554e-05 <= x < 0.0010953:
        return 2
    elif -0.00151104 <= x < -9.53554e-05:
        return 1
    else:
        return 0


def happiness_label(x):
    if x >= 0.00699622:
        return 4
    elif 0.00167327 <= x < 0.00699622:
        return 3
    elif -0.00180106 <= x < 0.00167327:
        return 2
    elif -0.00682289 <= x < -0.00180106:
        return 1
    else:
        return 0


def credence_label(x):
    if x >= 0.0142478:
        return 4
    elif 0.00306959 <= x < 0.0142478:
        return 3
    elif -0.00106789 <= x < 0.00306959:
        return 2
    elif -0.00716395 <= x < -0.00106789:
        return 1
    else:
        return 0


def hope_label(x):
    if x >= 0.00749548:
        return 4
    elif 0.00161387 <= x < 0.00749548:
        return 3
    elif -0.00123102 <= x < 0.00161387:
        return 2
    elif -0.00393997 <= x < -0.00123102:
        return 1
    else:
        return 0
