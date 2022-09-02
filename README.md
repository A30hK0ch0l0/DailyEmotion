# Install

`PROJECT_DIR` is project root

```bash
mkdir ${PROJECT_DIR}
cd ${PROJECT_DIR}

sudo apt update -y
sudo apt install -y python3 git 
sudo apt install -y python3-pip
python3 -m pip install virtualenv

git clone http://185.208.77.246/inference/daily_emotion.git .
git checkout develop

python3 -m virtualenv venv
source venv/bin/activate
pip install -U pip wheel setuptools pytest
pip install -r requirements.txt
```

# Test

```bash
bash ./bin/test.sh

# or

source venv/bin/activate
pytest
```

## Install package for other projects

```bash
pip install git+http://185.208.77.246/inference/daily_emotion.git
```

# How to use emotion twitter

Change `text` variable

```python
from copy import deepcopy
import pandas as pd
from daily_emotions.days3_prediction import days3

# emotion values example
emotion_dic = {
    'time': ['2021-11-07'] * 10,
    'anger': [0.04178459062176727] * 10,
    'credence': [0.04178459062176727] * 10,
    'expectation': [0.04178459062176727] * 10,
    'fear': [0.04178459062176727] * 10,
    'happiness': [0.04178459062176727] * 10,
    'hate': [0.04178459062176727] * 10,
    'hope': [0.04178459062176727] * 10,
    'sadness': [0.04178459062176727] * 10,
    'wonder': [0.04178459062176727] * 10,
}

print(emotion_dic)
# cast to dataframe
emotion_dataframe = pd.DataFrame(emotion_dic)

# predict
print(days3(emotion_dataframe))

```

# How to use emotion instagram

Change `text` variable

```python
from copy import deepcopy
import pandas as pd
from daily_emotions_instagram.prediction_3days import days3

# emotion values example
emotion_dic = {
        'time': ['1400-08-05'] * 10,
        'anger': [0.028787645914396885] * 10,
        'credence': [0.05468730544747082] * 10,
        'expectation': [0.0973669747081712] * 10,
        'fear': [0.0006862354085603112] * 10,
        'happiness': [0.04710530155642023] * 10,
        'hate': [0.0025732003891050584] * 10,
        'hope': [0.037002966926070045] * 10,
        'sadness': [0.01980627431906615] * 10,
        'wonder': [0.009900437743190662] * 10,
    }


print(emotion_dic)
# cast to dataframe
emotion_dataframe = pd.DataFrame(emotion_dic)

# predict
print(days3(emotion_dataframe))

```
