
#####PROCESAMIENTO#####
def get_bounds(x,q_down:float=0.05,q_up:float=0.95,factor:float=1.5):
    '''Obtener intervalo de aceptación para no considerar outliers'''
    q3 = x.quantile(q_up)
    q1 = x.quantile(q_down)
    iqr = q3 - q1
    lb = q1 - factor*iqr
    ub = q3 + factor*iqr
    return pd.Interval(lb, ub, closed="both")

def distancia_xy(x,y):
    '''Distancia en km entre dos puntos'''
    dis = distance.distance(x, y).km
    return dis

def rango_km(df,fecha,latitud,longitud,n_weeks):
    '''Determinar si un punto está dentro del rango de '''
    fechaini = (datetime.strptime(fecha,'%Y-%m')-timedelta(weeks=n_weeks)).strftime('%Y-%m')
    aux = df[(df.year_month<=fecha) & (df.year_month>=fechaini)].copy()
    resultados = [distancia_xy((latitud,longitud),(latitud_obs,longitud_obs)) 
                  for latitud_obs,longitud_obs in zip(aux.lat,aux.long)]
    return np.median(resultados)

def address(coordinates):
    '''Obtener dirección según las coordenadas'''
    location = locator.reverse(coordinates)
    return location.raw['address']

#####MODELO#####

def classification_metrics(X, y, model,pipe=None,scores:tuple=('roc_auc')):
    '''Medir performance del modelo'''
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", model)])
    else:
        pipe = Pipeline([("model", model)])
    ls_scores = cross_val_score(estimator=pipe, X=X, y=y, scoring=scores, n_jobs=-1, cv=4)
    print(f"Media: {np.mean(ls_scores):,.2f}, STD: {np.std(ls_scores)}")

def hyperparam_logistic(X, y, pipe=None):
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", LogisticRegression())])
    else:
        pipe = Pipeline([("model", LogisticRegression())])
    param_grid = {"model__penalty": ["l1", "l2"],
                "model__C": [x/100 for x in range(100)]+[0],
                "model__class_weight": ["balanced", None],
                "model__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                }
    hp = RandomizedSearchCV(cv=4, 
                          param_distributions=param_grid,
                          n_iter=150,
                          scoring="roc_auc", 
                          verbose=10,
                          error_score=-1000, 
                          estimator=pipe, 
                          n_jobs=-1,
                          random_state=0)
    hp.fit(X=X, y = y)
    print(f"ROC: {hp.best_score_:,.2f}")
    return hp

def hyperparam_neural(X, y, pipe=None):
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", MLPClassifier())])
    else:
        pipe = Pipeline([("model", MLPClassifier())])
    param_grid = {"model__hidden_layer_sizes": [(a,b,c,) for a in range(10,60,10) for b in range(10,60,10) for c in range(10,60,10)],
                "model__activation": ['logistic', 'tanh', 'relu'],
                "model__solver": ['lbfgs', 'sgd', 'adam'],
                "model__alpha": np.arange(0.01,1,0.01),
                "model__learning_rate": ['constant', 'invscaling', 'adaptive'],
                }
    hp = RandomizedSearchCV(cv=4, 
                          param_distributions=param_grid,
                          n_iter=150,
                          scoring="roc_auc", 
                          verbose=10,
                          error_score=-1000, 
                          estimator=pipe, 
                          n_jobs=-1,
                          random_state=0)
    hp.fit(X=X, y = y)
    print(f"ROC: {hp.best_score_:,.2f}")
    return hp

def hyperparam_svc(X, y, pipe=None):
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", SVC())])
    else:
        pipe = Pipeline([("model", SVC())])
    param_grid = {"model__C": np.arange(0.1,2,0.1),
                "model__kernel": ['linear', 'rbf', 'sigmoid','poly'],
                "model__degree": range(2,5),
                "model__probability": [True],
                }
    hp = RandomizedSearchCV(cv=4, 
                          param_distributions=param_grid,
                          n_iter=150,
                          scoring="roc_auc", 
                          verbose=10,
                          error_score=-1000, 
                          estimator=pipe, 
                          n_jobs=-1,
                          random_state=0)
    hp.fit(X=X, y = y)
    print(f"ROC: {hp.best_score_:,.2f}")
    return hp

#####VISUALIZACIONES#####
def bar_periods(df,columna,title):
    aux = df[columna].value_counts().to_frame().sort_index()
    plt.subplots(figsize=(18,8))
    plt.title(title)
    aux[columna].plot(kind='bar', color='violet')
    plt.xticks(rotation=90, fontsize=15)
    plt.show()