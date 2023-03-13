
import geopandas as gpd
import pandas as pd 
import numpy as np 
import pyspatialml
import os
import xgboost
import time 
import rasterio
import catboost
import pickle


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from glob import glob 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,PowerTransformer,Normalizer
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

#[] dartington code 




vnames = ['zdif','lidar','tdx','cop','merit','nasa','aw3d','egm96', 'ethm','eths',
          'fnf','esawc','wsf', 's1vv','s1vh','s1a','s2r','s2g','s2b']

namefiles = ['zdif','lidar','tdx','cop','merit','nasa','aw3d','egm96', 'ethm','eths',
             'fnf','esawc','wsf', 's1','s2']

######################################################################
############################ IO function  ############################
######################################################################

def pickle_load_model(model_path):
    pickled_model = pickle.load(open(model_path, 'rb'))
    return pickled_model

def pickle_write_model(model, model_path):
    pickle.dump(model, open(model_path, 'wb'))
    return model_path

######################################################################
############################ PREPROCESSING ###########################
######################################################################

def scale_data(xdf, method='StandardScaler'):
    if method == 'StandardScaler': scaler = StandardScaler()
    elif method == 'MinMaxScaler': scaler = MinMaxScaler()
    elif method == 'RobustScaler': scaler = RobustScaler()
    elif method == 'Normalizer': scaler = Normalizer()
    ### add more
    dfx = scaler.fit_transform(xdf)
    return dfx

def process_dummy(): # categotical 
    pass

def df_train_valid_test_split(df, frac=0.25):
    # same can be done with train_test_split from sklearn 
    # shuffle the DataFrame rows
    df = df.sample(frac = 1)

    # get random sample 
    valid = df.sample(frac=frac, axis=0)

    # get everything but the test sample
    train = df.drop(index=valid.index)

    test = valid.sample(frac=0.1, axis=0)
    valid = valid.drop(index=test.index)

    print(f'train : {train.shape} valid:{valid.shape} test:{test.shape}')

    return train, valid, test 

def split_train_valid(df, tsize=0.15):
    trainx, validx, trainy, validy = train_test_split(df.drop(['zdif'], axis=1), 
                                                  df['zdif'],
                                                  shuffle= True,
                                                  test_size=tsize)


    print(trainx.shape, trainy.shape)
    print(validx.shape, validy.shape)
    return trainx, validx, trainy, validy 
######################################################################
################## SAMPLING AND RESAMPLING ###########################
######################################################################
def cv_strategy(cv, cfl, X,y,outdir,name='xbg_loo'):
    ti = time.perf_counter()
    kfoldlist = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f'############ {i} ##############')
        xtrain, xtest = X[train_idx], X[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx] 

        #model_path = f'{model_dir}/XGB_kfold{i}.json'
        model_dir = f'{outdir}/{name}'
        os.makedirs(model_dir, exist_ok=True)
        model_path = f'{model_dir}/{name}_fold{i}.sav'

        model, eval = train_predict(cfl, xtrain,ytrain, xtest, ytest)
        print(eval[0], eval[1])

        pickle_write_model(model, model_path)
        #model.save_model(model_path)
        #pickle.dump(model, open(model_path, "wb"))

        kfoldlist.append({
            'kfold':i, 'model':model_path,
            'rmse':eval[0], 'r2':eval[1],
            'nsample':xtrain.shape[0],
            'perc':(xtrain.shape[0]/X.shape[0]) * 100
        })

    
    tf = time.perf_counter() - ti 
    tf = round(tf/60,3)
    fn = f'{model_dir}/{name}_time{tf}mins.csv'
    df = pd.DataFrame(kfoldlist).sort_values('rmse')
    df.to_csv(fn, index=False)
    print(fn)
    return df 

def spatial_groupkfoldcv(X:np.array, y:np.array, kclusters,
                        name:str, outdir:str, clf,cv,c):

    ti = time.perf_counter()
    kfoldlist = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y, kclusters)):
        print(f'############ {i} ##############')
        xtrain, xtest = X[train_idx], X[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx] 

        #model_path = f'{model_dir}/XGB_kfold{i}.json'
        model_dir = f'{outdir}/{name}'
        os.makedirs(model_dir, exist_ok=True)
        model_path = f'{model_dir}/{name}_fold{i}.sav'

        model, eval = train_predict(clf, xtrain,ytrain, xtest, ytest)
        print(eval[0], eval[1])

        pickle_write_model(model, model_path)
        #model.save_model(model_path)
        #pickle.dump(model, open(model_path, "wb"))

        kfoldlist.append({
            'kfold':i, 'model':model_path,
            'rmse':eval[0], 'r2':eval[1],
            'nsample':xtrain.shape[0],
            'perc':(xtrain.shape[0]/X.shape[0]) * 100
        })

    tf = time.perf_counter() - ti 
    tf = str(round(tf/60,3)).replace('.','p')
    fn = f'{model_dir}/{name}_c{c}_time{tf}mins.csv'
    df = pd.DataFrame(kfoldlist).sort_values('rmse')
    df.to_csv(fn, index=False)
    print('run.time', tf)
    return df





######################################################################
############################ PERFORMANCE EVAL ########################
######################################################################

def eval_model(model, X,y):
    # write these into a file 
    p = model.predict(X)
    perf_eval(y,p)


def perf_eval(y, p):
    rmse = mean_squared_error(y,p, squared=False)
    r2   = r2_score(y, p)
    print(f'rmse:{rmse} r2:{r2}')
    return [rmse, r2]


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,explained_variance_score,median_absolute_error

def regression_metrics(y_true, y_pred):
    """
    Calculates common regression metrics using scikit-learn.
    Arguments:
        y_true: array-like, true target values
        y_pred: array-like, predicted target values
    Returns:
        A dictionary of metric names and their corresponding values.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    evs = explained_variance_score(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    out = {'R^2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'EVS': evs, 'MeAE': mdae}
    out_scores = pd.DataFrame(out, index=[0])

    print(out_scores)
    return out_scores


def EVAL_MODEL(model, X,y):
    # write these into a file 
    p = model.predict(X)
    regression_metrics(y,p)

######################################################################
############################ TRAIN AND PREDICT ###########################
######################################################################

def fit_catboost(trainx, validx, trainy, validy,model_path):
    ## make it generalizeble []

    early_stop = 100
    floss  = 'RMSE'
    lrate = 0.1
    model = CatBoostRegressor(iterations=10000, verbose=100, 
                                early_stopping_rounds=early_stop, 
                                loss_function= floss,
                                depth = 6, 
                                learning_rate=lrate,
                                task_type="GPU", devices="0:1")
    model.fit(trainx, trainy, eval_set=(validx, validy))

    pickle_write_model(model, model_path)
    
    return model




######################################################################
############################ FEATURES IMP AND SEL ####################
######################################################################


########################################################################
def train_predict(model, trainx,trainy, testx, testy):
    ti = time.perf_counter()
    model.fit(trainx, trainy)
    predy = model.predict(testx)

    # Dartington functions
    eval = perf_eval(testy, predy)
    # save model

    tf = time.perf_counter() - ti
    print(f'run.time = {tf/60} mins')

    return model, eval

def get_clusters(X:pd.DataFrame, c=12):
    # make it agnostic
    coords = X[['x','y']]
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)
    kmeans = KMeans(
            init='random', n_clusters=c,
            n_init='auto', max_iter=300,
            random_state=42)
    kmeans.fit(scaled_coords)
    kclusters = kmeans.labels_
    return kclusters

def df_to_path_generator(df):
    varspath_list = [[cop, nasa, aw3d, lgeoid,egm96, egm08, lidar,tdemx,esri] \
        for cop, nasa, aw3d, lgeoid,egm96,egm08,lidar,tdemx,esri in \
            zip(df['cop'], df['nasa'],df['aw3d'],df['lgeoid'],df['egm96'],df['egm08'],df['lidar'], df['tdemx'] ,df['esri'])]
    print(len(varspath_list))
    return varspath_list


def read_oneband(path):
    with rasterio.open(path) as ds:
        r = ds.read(1)
       # print(r.shape)
    return r

def read_multiband(path):
    with rasterio.open(path) as ds:
        r = ds.read()
       # print(r.shape)
    return r

def process_mask(mask,uthresh,lthresh, nval = -9999):
    mask_temp = mask.copy()
    mask_temp[mask <=-lthresh] = nval
    mask_temp[mask >= uthresh] = nval
    return mask_temp