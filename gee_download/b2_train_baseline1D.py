from glob import glob 
from UtilsFunc import get_idx_batch_paths,generate_random_indices, load_data_byindices,writepath2txt
from UtilsFunc import get_current_datetime_uk, set_np_seed
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,PowerTransformer,Normalizer
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
### https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
import pandas as pd 
import numpy as np 
import os 
import pickle


from UtilsML import scale_data,split_train_valid,fit_catboost,eval_model,EVAL_MODEL

stime = get_current_datetime_uk()
###################################################
# training experiement # keep tile [1] untouched 
# v1: sample train tile A | predict tile A 
# v2: sample train tile A | predict tile B > do the sample for A,B,C,D
# v3: sample train tiles A,B,C | predict B > each permutation 
# v4: sample region - train all tiles and predict all tiles 

Kpct = 0.07#0.1 # multiplying factor the min and max cop to threshold tdx : UW 
Nsize = 1001
path = '/media/ljp238/6tb/Joseph/DATASETS/OUT_TILES/1x1degree/'

acols = ['x', 'y', 'zdif', 'lidar', 'tdx', 'cop', 'merit', 'nasa', 'aw3d', 's2r','s2g', 's2b', 'id']
rcv_cols = ['zdif', 'cop', 'merit', 'nasa', 'aw3d', 's2r','s2g', 's2b']

if __name__ =='__main__':

    set_np_seed()

    print('Loading the patch_paths.csv')
    fs = sorted(glob(f'{path}/*//**//*_patches_paths.csv', recursive=True))
    csv_tile_patches_path  = fs[-2] #@

    print('Setting the folders')
    R, H = os.path.split(csv_tile_patches_path)
    outproc = os.path.join(R, 'outproc')
    os.makedirs(outproc, exist_ok=True)
    train_sample_dirpath = os.path.join(outproc, f'Train{Nsize}patches')
    os.makedirs(train_sample_dirpath, exist_ok=True)

    print(len(fs))

    print('Generating Indices from patches')

    t = pd.read_csv(csv_tile_patches_path)
    train_indices,valid_indices,test_indices = generate_random_indices(t,Nsize)

   # idx = 100

    ## [][]concat dfs for csv_tile_patches_path before generating sample or do it latter each
   # idx_patch_paths = get_idx_batch_paths(t,idx)
    
    txt_train_indices = os.path.join(train_sample_dirpath, 'train_indices.txt')
    txt_valid_indices = os.path.join(train_sample_dirpath, 'valid_indices.txt')
    txt_test_indices = os.path.join(train_sample_dirpath, 'test_indices.txt')
    writepath2txt(txt_train_indices, train_indices)
    writepath2txt(txt_valid_indices, valid_indices) 
    writepath2txt(txt_test_indices, test_indices)

    print('loading all the patches @# bottlenect > replace by dask')
    traindf = load_data_byindices(train_indices,t, train_sample_dirpath)

    # [] plot boxplot and describe across all variables
    print('Removing invalid data')
    df = traindf[rcv_cols]
    uthresh =  np.max(df.cop) + np.max(df.cop) * Kpct
    lthresh =  np.min(df.cop) - np.min(df.cop) * Kpct
    df = df[df.zdif >= lthresh] # use copernicus to filter low and high values or lidar ## apply to 2D too
    df = df[df.zdif <= uthresh] # 100
    dx = df.copy()
    dx = dx.dropna()
     # [] plot boxplot and describe across all variables

    print('Preprocessing the data numerical only for now')

    trainx, validx, trainy, validy = split_train_valid(dx)
    # [] plot boxplot and describe across all variables


    trainx_s = scale_data(trainx, 'StandardScaler')
    trainx_m = scale_data(trainx, 'MinMaxScaler')
    trainx_r = scale_data(trainx, 'RobustScaler')
    trainx_n = scale_data(trainx, 'Normalizer')

    validx_s = scale_data(validx, 'StandardScaler')
    validx_m = scale_data(validx, 'MinMaxScaler')
    validx_r = scale_data(validx, 'RobustScaler')
    validx_n = scale_data(validx, 'Normalizer')

    print('data  resampling strategy')


    print('Fitting the models...')
    modeldir = os.path.join(train_sample_dirpath, 'models')
    os.makedirs(modeldir, exist_ok=True)
    
    # [] write history data into csv and plot latter 

    modelpath_s = os.path.join(modeldir, f'cboost_s_{get_current_datetime_uk()}{Nsize}.pkl')
    model_s = fit_catboost(trainx_s, validx_s, trainy, validy,modelpath_s)

    modelpath_m = os.path.join(modeldir, f'cboost_m_{get_current_datetime_uk()}{Nsize}.pkl')
    model_m = fit_catboost(trainx_m, validx_m, trainy, validy,modelpath_m)

    modelpath_r = os.path.join(modeldir, f'cboost_r_{get_current_datetime_uk()}{Nsize}.pkl')
    model_r = fit_catboost(trainx_r, validx_r, trainy, validy,modelpath_r)

    modelpath_n = os.path.join(modeldir, f'cboost_n_{get_current_datetime_uk()}{Nsize}.pkl')
    model_n = fit_catboost(trainx_n, validx_n, trainy, validy,modelpath_n)

    print('Evaluating the models the models...')
    # [] write the errors and plot dist to disk 
    # [] plot distribution of errors
    EVAL_MODEL(model_s, trainx_s, trainy)
    EVAL_MODEL(model_s, validx_s, validy)
    print('')

    EVAL_MODEL(model_m, trainx_m, trainy)
    EVAL_MODEL(model_m, validx_m, validy)
    print('')

    EVAL_MODEL(model_r, trainx_r, trainy)
    EVAL_MODEL(model_r, validx_r, validy)
    print('')

    EVAL_MODEL(model_n, trainx_n, trainy)
    EVAL_MODEL(model_n, validx_n, validy)
    print('')

    print('Modell Performance Eval history plots')


    print('Modell something - feature impotance ')


    print('Modell something - area of applicability and dissimilarity index')


    print(f'lthresh={lthresh} >> {np.min(df.cop)} uthresh={uthresh} >> {np.max(df.cop)}')
    print(dx.isna().sum())
    print(f'df ntrhesh: {df.shape} df wthresh: {dx.shape} >> {dx.shape[0]/df.shape[0]}')



    ftime = get_current_datetime_uk()
    print(f'stime: {stime} \netime: {ftime}')







