import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score

from modelo.UFO_model import UFOPredict

import pickle

path = 'datos/ufo_data.csv'
columns = ['occurance', 'city', 'citystate', 'country', 'ufoshape', 'dseconds', 'duration', 'description', 'report_date', 'lat','long']
ufo_data = pd.read_csv(path, sep=',', names =columns, low_memory=False)
ufo_data = ufo_data.drop(0, axis=0)
ufo_data_validation = ufo_data.head(int(len(ufo_data)*0.3)).copy()
ufo_data_dev = ufo_data.loc[~ufo_data.index.isin(ufo_data_validation.index)].copy()
name_pkl = 'moodelo.pkl'

predict = False
if not predict:
    modelo = UFOPredict(ufo_data_dev)
    modelo.processing(ufo_data_dev)
    modelo.train()
    pickle.dump(modelo,open(name_pkl,'wb'))
predict = True
modelo = pickle.load(open(name_pkl,'rb'))
tad = modelo.processing(ufo_data_validation,predict='validate')
tad = tad[modelo.best_features].copy()
y_predict = modelo.model.predict(tad[modelo.best_features])
print('ROC auc validate: ',roc_auc_score(modelo.y['validate'],y_predict))
modelo