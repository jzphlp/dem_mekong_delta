import ee 
import os 
import geemap
import geopandas as gpd 
import numpy as np
import time
from glob import glob
from UtilsGEE import * 


VAR ='EDM' #'WBM'[1] EDM FLM HEM DEM : how do you edit dem so it's seamless? github guys,and documentation???
gpkg_out = '/media/ljp238/6tb/Joseph/DATASETS/mekong_delta_data_tiles/wdir/1x1degree/gpkg_patches'
os.makedirs(gpkg_out, exist_ok=True)
data_path  = '/media/ljp238/6tb/Joseph/DATASETS/mekong_delta_data_tiles/wdir/1x1degree/'
outdir = '/media/ljp238/6tb/Joseph/DATASETS/COPDEM/'
CPUS = int(os.cpu_count() * 0.9)
SCALE = 30
if __name__ == '__main__':

    fs = glob(f'{data_path}/*/*.tif'); print(len(fs))
    tdxs = [i for i in fs if 'TANDEMX.tif' in i]
    gpkgs = [os.path.join(gpkg_out, os.path.basename(i).replace('.tif', '.gpkg')) for i in tdxs ]
    print(tdxs[0],gpkgs[0])

    print('Checking that gpkg tile have been generated :@@')
    for i in range(len(tdxs)):
        gpkg_tile = os.path.join(gpkg_out, os.path.basename(tdxs[i]).replace('.tif',''))
        geotile_generate_tiles(tdxs[i], gpkgs[i], gpkg_tile)

    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except:
        ee.Authenticate()
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    with ThreadPoolExecutor(CPUS) as PEX:
        for fi in range(len(gpkgs)):
            #if fi > 0: break
            gfile = gpkgs[fi]
            g = gpd.read_file(gfile)
            g[['minx','miny','maxx','maxy']] = g.bounds
            #print(gfile)
            for i in range(g.shape[0]):
                roi, fname = get_ee_geometry(i,g, VAR)
                dem,edm,flm,hem,wbm = getDEM_files(roi)
                tiledir_var = os.path.join(outdir,os.path.basename(gfile).replace('_TANDEMX.gpkg', ''), VAR)
                os.makedirs(tiledir_var, exist_ok=True)
                outpath = os.path.join(tiledir_var, fname)
                if VAR == 'DEM':img = dem
                elif VAR == 'EDM':img = edm
                elif VAR == 'FLM':img = flm
                elif VAR == 'HEM':img = hem
                elif VAR == 'WBM':img = wbm
                
                PEX.submit(gee_download_geemap,img,outpath, SCALE)
    
    
    print('The End')

