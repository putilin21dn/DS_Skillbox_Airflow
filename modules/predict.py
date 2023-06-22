import json

import dill
import os
from datetime import datetime
import pandas as pd
def pred(file, model):
    df = pd.DataFrame.from_dict([file])
    y = model['model'].predict(df)
    d = {'car_id' : df['id'], 'pred' : [y[0]]}
    return pd.DataFrame(data=d)

def predict():

    path = os.environ.get('PROJECT_PATH', '..')
    models = os.listdir(f'{path}/data/models')

    with open(f'{path}/data/models/{models[len(models)-1]}', 'rb') as m:
        model = dill.load(m)

    list = []

    for file in os.listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/{file}') as f:
            fin = json.load(f)
        list.append(pred(fin, model))
    df = pd.concat(list)
    df_prev = df.set_index('car_id')
    print(df_prev)
    with open(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', 'wb') as csv:
        df_prev.to_csv(csv)
if __name__ == '__main__':
    predict()
