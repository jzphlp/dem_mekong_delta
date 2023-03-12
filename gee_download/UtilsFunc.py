import os
import tarfile
import shutil
from glob import glob

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
            T.write(fi+'\n')

############################################################# all funtions up to go Utilstilling



