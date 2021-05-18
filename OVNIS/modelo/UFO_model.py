import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from modelo.model_selection import classification_metrics,hyperparam_logistic

from modelo.processing import tipo_datos,missings,outliers,feature_engineering, create_tad, escalamiento, feature_selection

class UFOPredict():
    def __init__(self,ufo_data):
            self.data = ufo_data 
            self.data_processing  = None
            self.ohe = None
            self.cv_description = None
            self.mm = None
            self.tad= None
            self.y = {}
            self.cols_ohe = None
    def processing(self,data,predict=False):
        data_processing = data.copy() 
        data_processing = tipo_datos(data_processing)
        data_processing = missings(data_processing)
        data_processing = feature_engineering(data_processing)
        data_processing = outliers(data_processing)
        ls_disc = ['country','day_occ', 'weekday_occ', 'month_occ','hour_occ']
        if not predict:
            ohe = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
            ohe.fit(data_processing[ls_disc])
            self.ohe = ohe
            cols_ohe = [x.replace('x0','country') if x.startswith('x0') else x.replace('x1','day_occ') if x.startswith('x1') 
            else x.replace('x2','weekday_occ') if x.startswith('x2') else x.replace('x3','month_occ') if x.startswith('x3') else
            x.replace('x4','hour_occ') for x in self.ohe.get_feature_names()]
            self.cols_ohe = cols_ohe
        data_processing[self.ohe.get_feature_names()] = self.ohe.transform(data_processing[ls_disc])
        data_processing.rename(columns=dict(zip(self.ohe.get_feature_names(),self.cols_ohe)), inplace=True)
        data_processing[[x for x in self.cols_ohe if x not in data_processing.columns]] = 0
        data_processing.drop(columns=ls_disc, inplace = True)
        if not predict:
            cv_description = CountVectorizer(stop_words=stopwords.words("english"), ngram_range=(1, 1), min_df=1, max_features=100)
            cv_description.fit(data_processing["description"])
            self.cv_description = cv_description
        desc = pd.DataFrame(data=self.cv_description.transform(data_processing["description"]).todense(), columns=[f"word_{x}" for x in self.cv_description.get_feature_names()])
        data_processing = data_processing.reset_index().join(desc)
        data_processing.drop([ 'description','index','citystate'], axis=1, inplace=True)
        if not predict:
            ls_unary = [x for x, y in data_processing.apply(lambda x: x.nunique()).items() if y == 1]
            data_processing.drop(columns=ls_unary, inplace=True)
            self.data_processing = data_processing
        tad = create_tad(data_processing,predict)
        tad = escalamiento(tad,self,predict)
        if not predict:
            self.best_features = feature_selection(tad,self)
            #self.best_features = ['word_light_sum', 'word_object_sum', 'word_nuforc_sum', 'word_east_sum','word_triangle_sum', 'days_toreport_median', 'word_orange_sum',
            #                    'word_silver_sum', 'country_us_sum', 'hour_occ_17_sum']
            self.tad = tad[self.best_features]
        if predict in [True,'validate']:
            return tad
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.tad, self.y['train'], test_size=0.3, random_state=777)
        print('Train: ',y_train.y.value_counts(1),'Test:\n',y_test.y.value_counts(1))
        hp_logistic = hyperparam_logistic(X_train,y_train)
        be_logistic = hp_logistic.best_estimator_.steps[0][1]
        self.roc_auc_train = roc_auc_score(y_train,be_logistic.predict(X_train))
        self.roc_auc_test = roc_auc_score(y_test,be_logistic.predict(X_test))
        print('ROC auc train: ',self.roc_auc_train,'\nROC auc test: ',self.roc_auc_test)
        self.model = be_logistic