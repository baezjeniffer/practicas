import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from varclushi import VarClusHi

def leer_archivo(path:str):
    columns = ['occurance', 'city', 'citystate', 'country', 'ufoshape', 'dseconds', 'duration', 'description', 'report_date', 'lat','long']
    ufo_data = pd.read_csv(path, sep=',', names =columns, low_memory=False)
    data = ufo_data.copy()
    data = data.drop(0, axis=0)
    return data

def get_bounds(x,q_down:float=0.05,q_up:float=0.95,factor:float=1.5):
    '''Obtener intervalo de aceptación para no considerar outliers'''
    q3 = x.quantile(q_up)
    q1 = x.quantile(q_down)
    iqr = q3 - q1
    lb = q1 - factor*iqr
    ub = q3 + factor*iqr
    return pd.Interval(lb, ub, closed="both")

def outliers(df,columnas_continuas:list=["dseconds","days_toreport"]):
    data = df.copy()
    dc_out = {feat: get_bounds(data[feat]) for feat in columnas_continuas}
    for col in columnas_continuas:
        data[f"ol_{col}"] = data[col].map(lambda x: x not in dc_out[col]).astype(int)
    data["ol"] = data[[x for x in data.columns if x.startswith("ol")]].mean(axis=1)
    data["ol"].describe([0.9, 0.95, 0.96, 0.97, 0.98, 0.99])
    data = data[data["ol"]<=0.3].drop(columns=[x for x in data.columns if x.startswith("ol")])
    return data 

def missings(df):
    data = df.copy()
    data.drop(data[data.duration.isnull()].index, inplace = True)
    data.drop(data[data.dseconds.isnull()].index, inplace = True)
    data.drop(data[data.lat.isnull()].index, inplace = True)
    data.drop(data[data.ufoshape.isnull()].index, inplace = True)
    data['country'] = data['country'].fillna('ND')
    data['citystate'] = data['citystate'].fillna('ND')
    data['description'] = data['description'].fillna('-')
    return data

def tipo_datos(df):
    data = df.copy() 
    data['occurance'] = data.occurance.str.replace('24:00', '00:00')
    data['occurance'] = pd.to_datetime(data['occurance'], format='%m/%d/%Y %H:%M')
    data['report_date'] = pd.to_datetime(data['report_date'], format='%m/%d/%Y', errors='coerce')
    data['dseconds'] = pd.to_numeric(data['dseconds'], errors = 'coerce')
    data['dseconds'] = pd.to_numeric(data['dseconds'], errors = 'coerce')
    data['lat'] = pd.to_numeric(data['lat'], errors = 'coerce')
    data['long'] = pd.to_numeric(data['long'], errors = 'coerce')
    return data

def escalamiento(df,modelo,predict=False):
    data = df.copy()
    if not predict:
        modelo.y['train'] = data[["y"]]
        y = modelo.y['train'].copy()
        mm = MinMaxScaler()
        X = data.drop(columns = ["y",'year_week_occ','ufoshape'], axis = 1)
        mm.fit(X)
        modelo.mm = mm
    elif predict in [True,'validate']:
        if predict == 'validate':
            modelo.y['validate'] = data[["y"]]
            y = modelo.y['validate'].copy()
            X = data.drop(columns = ["y",'year_week_occ','ufoshape'], axis = 1)
        else:
            X = data.drop(columns = ['year_week_occ','ufoshape'], axis = 1)
        mm = modelo.mm
    Xmm = pd.DataFrame(index = X.index, data = mm.transform(X), columns=X.columns) 
    return Xmm

def feature_engineering(df,cut_normalize:float=0.005):
    data = df.copy()
    data['country'] = data['country'].apply(lambda x: 'OTRO' if x in ['ca','gb','au','de'] else x)
    aux = (data.ufoshape.value_counts(1)<cut_normalize).to_frame()
    ls_ufoshape_normalize = aux[aux.ufoshape==True].index.tolist()
    data['ufoshape'] = data.ufoshape.apply(lambda x: 'unpopular' if x in ls_ufoshape_normalize else x)
    data['day_occ'] = data['occurance'].dt.day
    data['weekday_occ'] = data['occurance'].dt.weekday
    data['month_occ'] = data['occurance'].dt.month
    data['year_occ'] = data['occurance'].dt.year
    data['year_week_occ'] = data['occurance'].apply(lambda x: x.strftime('%Y-%W'))
    data['hour_occ'] = data.occurance.apply(lambda x: x.strftime('%H'))
    data['occurance'] = data['occurance'].apply(lambda x: datetime(x.year,x.month,x.day))
    data['days_toreport'] = list(map(lambda x,y: (x-y).days,data['report_date'],data['occurance']))
    data = data[data.days_toreport>=0].reset_index(drop=True).copy()
    data['n_words'] = data.description.apply(lambda x: len(x.split(' ')))
    data['n_letrers'] = data.description.apply(lambda x: len(x))
    return data

def create_tad(df,predict=False):
    data = df.copy()
    data['year_month_occ'] = data.occurance.apply(lambda x: x.strftime('%Y-%m'))
    agg_func = {'dseconds':['median','std'],'days_toreport':['median']}
    desc_columns = [x for x in data.columns if x.startswith('word')]
    agg_func.update(dict(zip(desc_columns,['sum']*len(desc_columns))))   
    ohe_fts = [x for x in data.columns if x.startswith('word') or x.startswith('country') or x.startswith('hour')]
    agg_func.update(dict(zip(ohe_fts,['sum']*len(ohe_fts))))   
    ohe_fts = [x for x in data.columns if x.startswith('month') or x.startswith('weekday')]
    agg_func.update(dict(zip(ohe_fts,['max']*len(ohe_fts))))   
    agg_func.update({'year_month_occ':['min','max']})
    tad = data.groupby(['year_week_occ','ufoshape']).agg(agg_func).reset_index()
    tad.columns = ['_'.join(x) if x[1]!='' else x[0] for x in tad.columns]
    primer_fecha = tad.year_month_occ_min.min()
    ultima_fecha = tad.year_month_occ_max.max()
    rango_fechas = pd.date_range(primer_fecha,ultima_fecha,freq='W').strftime('%Y-%W')
    tad.drop(columns=['year_month_occ_min','year_month_occ_max'],inplace=True)
    ls_dfs = []
    for shape in data.ufoshape.unique():
        aux = tad[tad.ufoshape==shape].copy()
        if predict in [False, 'validate']:
            aux['y'] = 1
        aux_missing = pd.DataFrame({'year_week_occ':[x for x in rango_fechas if x not in aux.year_week_occ.tolist()]})
        if len(aux_missing)>0:
            if predict in [False, 'validate']:
                aux_missing['y'] = 0
            aux = aux.append(aux_missing)
        aux['ufoshape'] = shape
        if predict in [False, 'validate']:
            aux['y'] = aux['y'].shift(-1)
            aux.dropna(subset=['y'],inplace=True)
        aux[[x for x in aux.columns if x not in ['year_week_occ','ufoshape']]] = aux[[x for x in aux.columns if x not in ['year_week_occ','ufoshape']]].fillna(0)
        ls_dfs.append(aux)
    tad = pd.concat(ls_dfs,sort=False).sort_values(by=['ufoshape','year_week_occ']).reset_index(drop=True)
    tad.dropna(inplace=True)
    return tad

def feature_selection(df,modelo,clusters:int=10,selection:str='kbest',n_features:int=20):
    print('Feature selection')
    data = df.copy()
    if selection == 'varclushi':
        vc = VarClusHi(df=data, feat_list=data.columns,maxclus=n_features)
        vc.varclus()
        res = vc.rsquare.sort_values(by=["Cluster", "RS_Ratio"]).groupby(["Cluster"]).first()
        bf = [x for x in res["Variable"]]
    elif selection == 'kbest':
        kb = SelectKBest(k="all")
        kb.fit(data, modelo.y['train'])
        kb_scores = pd.DataFrame({'variable':data.columns,'score':kb.scores_})
        kb_scores = kb_scores.sort_values(by='score').reset_index(drop=True).dropna()
        bf = [x for x in kb_scores["variable"].tail(n_features)]
    return bf