import ee 
import os 
import geemap
import geopandas as gpd 
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from UtilsGEE import * 

try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
except:
    ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


gpkg_path = '/media/ljp238/6tb/Joseph/DATASETS/mekong_delta_data_tiles/wdir/1x1degree/gpkg_patches'
sentinel1_dir ='/media/ljp238/6tb/Joseph/DATASETS/SATIMG/SENTINEL1/'
cpus = int(os.cpu_count() * 0.9)
pol = 'VV'
scale = 10
i = 12
name = 'S1_VVVHangle'
if __name__ == '__main__':
    ti = time.perf_counter()
    gpkg_files = glob(f'{gpkg_path}/*.gpkg')
    #print(os.listdir(gpkg_path))
    print(gpkg_files)
    with ThreadPoolExecutor(cpus) as TEX:
        for fi in range(len(gpkg_files)):
            #if fi > 0: break
        
            gfile = gpkg_files[fi]
            g = gpd.read_file(gfile)
            g[['minx','miny','maxx','maxy']] = g.bounds
            print('gfile', gfile)

            tname = os.path.basename(gfile).replace('_TANDEMX.gpkg','')
            S1tile_path = os.path.join(sentinel1_dir, tname)
            os.makedirs(S1tile_path, exist_ok=True)
            for i in range(g.shape[0]):
                #if i > 30: break
                print(i)
               # getS1patch(i,g,pol,name,S1tile_path,scale)

                TEX.submit(getS1patch,i,g,pol,name,S1tile_path,scale)

    tf = time.perf_counter() - ti
    print(f'run.time {tf/60} min (s)')


