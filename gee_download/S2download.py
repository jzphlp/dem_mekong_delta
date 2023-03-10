import ee 
import os 
import geopandas as gpd 
import time
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from UtilsGEE import *
try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
except:
    ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


gpkg_path = '/media/ljp238/6tb/Joseph/DATASETS/mekong_delta_data_tiles/wdir/1x1degree/gpkg_patches'
sentinel2_dir ='/media/ljp238/6tb/Joseph/DATASETS/SATIMG/SENTINEL2/'
cpus = int(os.cpu_count() * 0.9)

name = 'S2_RGB'
scale = 10 
if __name__ == '__main__':
    ti = time.perf_counter()
    os.makedirs(sentinel2_dir, exist_ok=True)
    gpkg_files = glob(f'{gpkg_path}/*.gpkg')

    print(gpkg_files)
    with ThreadPoolExecutor(cpus) as TEX:
        for fi in range(len(gpkg_files)):
            #if fi > 0: break
        
            gfile = gpkg_files[fi]
            g = gpd.read_file(gfile)
            g[['minx','miny','maxx','maxy']] = g.bounds
            print('gfile', gfile)

            tname = os.path.basename(gfile).replace('_TANDEMX.gpkg','')
            S2tile_path = os.path.join(sentinel2_dir, tname)
            os.makedirs(S2tile_path, exist_ok=True)
            for i in range(g.shape[0]):
                #if i > 30: break
                print(i)
               # getS2RGBpatch(i,g,name,S2tile_path,scale)

                TEX.submit(getS2RGBpatch,i,g,name,S2tile_path,scale)

    tf = time.perf_counter() - ti
    print(f'run.time {tf/60} min (s)')