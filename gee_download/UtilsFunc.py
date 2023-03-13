import os
import tarfile
import shutil
import pyspatialml
import pandas as pd 
import numpy as np 
import datetime
import pytz

from glob import glob

def get_current_datetime_uk():
    # Set the timezone to UK
    uk_tz = pytz.timezone('Europe/London')

    # Get the current datetime in UK timezone
    now = datetime.datetime.now(uk_tz)

    # Format the datetime as a string
    date_str = now.strftime('%Y-%m-%d %H:%M:%S %Z%z')

    # Return the date and time as a list
    a = date_str.split()
   # dt = f'{a[0]}_{a[1]}_{a[2]}'.replace(':','-')
    dt = f'{a[0]}_{a[1]}'.replace(':','-').replace('-','')
    return dt


def cp(fi, fo):
    shutil.copyfile(fi, fo)
    print('Done!!!')

def filterlist(flist, k):
    f = [i for i in flist if k in i]
    f = f[0]
    return f

def dirlistfiles(dpath, ext, recursive=False):
    if not recursive:
        fs = sorted(glob(f'{dpath}/*.{ext}'))
        print('only main directory')
    elif recursive:
        print('subdirectiries included')
        fs = sorted(glob(f'{dpath}//**//*.{ext}', recursive=True))

    print(len(fs))
    return fs

def files_filter_list(flist, k):
    f = [i for i in flist if k in i]
    print(len(f))
    return f


def tarfile_extractall(tar_path,outdir):
    import tarfile, os
   # head, tail = os.path.split(tar_path)
    #outdir = os.path.join(head,tail[:-4])
    #os.makedirs(outdir, exist_ok=True)
    
    with tarfile.open(tar_path, 'r') as tf:
        tf.extractall(path=outdir)
    print('All files extracted')

def buildVRT_bypath(vrt, path):
    if not os.path.isfile(vrt):
        cmd = f'gdalbuildvrt {vrt} {os.path.join(path)}/*.tif'
        os.system(cmd)
    return vrt 

def buildVRT_bytxt(txt, vrt):
    if not os.path.isfile(vrt):
        # -a_srs  ESPG: UTMZONE or Global
        cmd = f'gdalbuildvrt -input_file_list {txt} {vrt}'
        # -vrtnodata -9999 
        os.system(cmd)
    print('done!!')

def gdal_vrt2tif(vrt,tif):
    # tif = vrt.replace('.vrt', '.tif') 
    cmd = f'gdal_translate {vrt} {tif}'
    os.system(cmd)
    return tif


def gdal_edit_ndv(path, ndv):
    # unsetnodata: remove exisiting nodata 
    # -a_nodata : assing a specific nodata value
    cmd = f'gdal_edit.py -a_nodata {ndv} {path}'
    os.system(cmd)


def writepath2txt(txt, fs):
    #fs = sorted(glob(f'{wdir}/*/*.tif', recursive=True))
    with open(txt, 'w') as T:
        for fi in fs:
            T.write(str(fi)+'\n')

def get_idx_batch_paths(t,idx):
    one_patch_paths = t.iloc[idx:idx+1,].values[0].tolist()
    return one_patch_paths

def set_np_seed():
    np.random.seed(42)

def generate_random_indices(t,N=180):
    import numpy as np 
    np.random.seed(t.shape[0])
    if N is not None:
        N = int(N + N*0.2)
        indices = np.random.randint(1, t.shape[0], size = N)
        print(len(indices))
    else:
        indices = np.random.randint(1, t.shape[0], size = t.shape[0])
        print(len(indices))

    total_length = len(indices)
    proportions = [0.7, 0.2, 0.1]
    sublist_lengths = [int(total_length * prop) for prop in proportions]
    train = indices[:sublist_lengths[0]]
    valid = indices[sublist_lengths[0]:sublist_lengths[0]+sublist_lengths[1]]
    test = indices[sublist_lengths[0]+sublist_lengths[1]:]
    train,valid,test = list(train), list(valid), list(test)

    
    print('train:', len(train))
    print('valid:', len(valid))
    print('test :', len(test))

    return train,valid,test

def get_idx_batch_paths_names_41D(idx_patch_paths):
    zdif_files = files_filter_list(idx_patch_paths, 'zdif')[0]
    lidar_files = files_filter_list(idx_patch_paths, 'LiDAR_MekongDelta_EGM08')[0]

    txd_files = files_filter_list(idx_patch_paths, 'TDX_DEM_EGM08')[0]
    cop_files = files_filter_list(idx_patch_paths, 'COP_DEM')[0]
    merit_files = files_filter_list(idx_patch_paths, 'MERIT_EGM08')[0]
    nasa_files = files_filter_list(idx_patch_paths, 'NASA_H_EGM08')[0]
    aw3d_files = files_filter_list(idx_patch_paths, 'AW3D_H_EGM08')[0]

    egm96_files = files_filter_list(idx_patch_paths, 'emg96')[0]
    s1_files = files_filter_list(idx_patch_paths, 'S1')[0]
    s2_files = files_filter_list(idx_patch_paths, 'S2')[0]

    ethm_files = files_filter_list(idx_patch_paths, 'ETHm')[0]
    eths_files = files_filter_list(idx_patch_paths, 'ETHs')[0]
    fnf_files = files_filter_list(idx_patch_paths, 'TDX_FNF')[0]
    esawc_files = files_filter_list(idx_patch_paths, 'WC')[0]
    wsf_files = files_filter_list(idx_patch_paths, 'WSF')[0]

    numpaths = [zdif_files,lidar_files,txd_files,cop_files,merit_files,
                 nasa_files,aw3d_files,s2_files]#,s1_files]
    numnames = ['zdif', 'lidar', 'tdx', 'cop', 'merit', 'nasa', 'aw3d', 's2r','s2g','s2b']#, 's1']
    return numnames, numpaths 


def write_idxpatch_to_dfparquet(t,idx, nsample_dirpath):
    idx_patch_paths = get_idx_batch_paths(t,idx)
    numnames, numpaths = get_idx_batch_paths_names_41D(idx_patch_paths)
    s = pyspatialml.Raster(numpaths)
    s.names = numnames
    bname = os.path.basename(numpaths[0])[:-4] + f'_idx{idx}.parquet'
    fparquet = os.path.join(nsample_dirpath, bname)
   # fgpkg = fparquet.replace('.parquet', '.gpkg')
    ds = s.to_pandas()
    ds['id'] = str(idx)
    ds.to_parquet(fparquet, index=False)
    del s 
    print(idx)
    print(fparquet)
    return ds

def load_data_byindices(indices,t, nsample_dirpath):
    dfs = [write_idxpatch_to_dfparquet(t,idx, nsample_dirpath) for idx in indices]
    di = pd.concat(dfs)
    return di 
############################################################# all funtions up to go Utilstilling



