from daily_emotions_instagram import data_path

#end_time = "1400-08-05"


emotion_mean_dic = {
    "anger": 0.028787645914396885,
    "wonder": 0.009900437743190662,
    "fear": 0.0006862354085603112,
    "hate": 0.0025732003891050584,
    "expectation": 0.0973669747081712,
    "sadness": 0.01980627431906615,
    "happiness": 0.04710530155642023,
    "credence": 0.05468730544747082,
    "hope": 0.037002966926070045
}

model_dic = {

    "anger": f"{data_path}/Data/01_anger.h5",
    "wonder": f"{data_path}/Data/02_wonder.h5",
    "fear": f"{data_path}/Data/03_fear.h5",
    "hate": f"{data_path}/Data/04_hate.h5",
    "expectation": f"{data_path}/Data/05_expectation.h5",
    "sadness": f"{data_path}/Data/06_sadness.h5",
    "happiness": f"{data_path}/Data/07_happiness.h5",
    "credence": f"{data_path}/Data/08_credence.h5",
    "hope": f"{data_path}/Data/09_hope.h5"
}

def anger_label(x):
    if x >= 0.002195:
        return 4
    elif 0.000297<= x <  0.002195:
        return 30.0
    elif -0.000539<= x < 0.000297:
        return 2
    elif  -0.002078 <= x < -0.000539:
        return 1
    else:
        return 0


def wonder_label(x):
    if x >= 0.000728:
        return 4
    elif 0.000198<= x < 0.000728:
        return 3
    elif -0.000206 <= x < 0.000198:
        return 2
    elif  -0.000737 <= x < -0.000206:
        return 1
    else:
        return 0


def fear_label(x):
    if x >=  0.000128:
        return 4
    elif 0.000013 <= x <  0.000128:
        return 3
    elif -0.000068 <= x < 0.000013:
        return 2
    elif  -0.000188 <= x < -0.000068:
        return 1
    else:
        return 0

def hate_label(x):
    if x >= 0.000517:
        return 4
    elif 0.000108 <= x < 0.000517:
        return 3
    elif -0.000122 <= x < 0.000108:
        return 2
    elif -0.000346 <= x < -0.000122:
        return 1
    else:
        return 0

def expectation_label(x):
    if x >=  0.013877:
        return 4
    elif 0.002287 <= x <  0.013877:
        return 3
    elif -0.002397 <= x < 0.002287:
        return 2
    elif -0.013635 <= x < -0.002397:
        return 1
    else:
        return 0

def sadness_label(x):
    if x >=    0.003007:
        return 4
    elif 0.000072 <= x <    0.003007:
        return 3
    elif -0.000900 <= x < 0.000072:
        return 2
    elif -0.002506 <= x < -0.000900:
        return 1
    else:
        return 0


def happiness_label(x):
    if x >=  0.010781:
        return 4
    elif 0.000312 <= x <  0.010781:
        return 3
    elif -0.002641 <= x < 0.000312:
        return 2
    elif -0.008071 <= x < -0.002641:
        return 1
    else:
        return 0

def credence_label(x):
    if x >=  0.004595:
        return 4
    elif 0.000405 <= x <  0.004595:
        return 3
    elif -0.001928 <= x < 0.000405:
        return 2
    elif -0.005929 <= x < -0.001928:
        return 1
    else:
        return 0

def hope_label(x):
    if x >=  0.005538:
        return 4
    elif  0.000893  <= x <  0.005538:
        return 3
    elif  -0.000926 <= x <   0.000893 :
        return 2
    elif -0.003495 <= x < -0.000926:
        return 1
    else:
        return 0

