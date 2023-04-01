from glob import glob
import os 
from geotile import GeoTile
from UtilsTiling import geotile_generate_tiles
from concurrent.futures import ProcessPoolExecutor
import time 

wx_patches_dir = ''
wdir = '/media/ljp238/6tb/Joseph/DATASETS/OUT_TILES/1x1degree/'
patch_size = 256
cpus = 50
# move this locally for faster readings
# change nameing convetion for Lgeoid and anyother tht needs changing
# uw the code and merge with  from GeoPatch import TrainPatch >>> Tuesday after presentation


if __name__ == '__main__':
    ti = time.perf_counter()
    with ProcessPoolExecutor(cpus) as PEX:
        tilenames = os.listdir(wdir)
        for i in range(len(tilenames)):
            #if i >= 1: break
            itile = tilenames[i]
            tile_path = os.path.join(wdir, itile)
            fs_tile = sorted(glob(f'{tile_path}/*.tif'))
            tile_patch_dir = os.path.join(tile_path, f'patch{patch_size}x{patch_size}')
            os.makedirs(tile_patch_dir, exist_ok=True)
            #print(tile_path)
            for tif_path in fs_tile:
                varname = itile + '_'+os.path.basename(tif_path).replace(itile, '')[1:].replace('.tif', '') #.lower()
                var_tile_patch_dir = os.path.join(tile_patch_dir, varname)
                os.makedirs(var_tile_patch_dir, exist_ok=True)
                gpkg_path = tif_path.replace('.tif', '.gpkg')
                gpkg_path = os.path.join(tile_patch_dir, os.path.basename(gpkg_path))
                print(f'file:{tif_path} \ndir:{var_tile_patch_dir} \ngpkg:{gpkg_path}')
                PEX.submit(geotile_generate_tiles, tif_path,gpkg_path,var_tile_patch_dir)




    tf = time.perf_counter() - ti 
    print(f'run.time {tf/60} min(s)')