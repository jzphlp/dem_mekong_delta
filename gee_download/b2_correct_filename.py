import pandas as pd 
from UtilsFunc import dirlistfiles,filterlist,files_filter_list
import os 
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from glob import glob

def correct_filenaming(fi):
    fo = fi.replace('..tif','.tif')
    os.rename(fi, fo)

pathdir = 'patch256x256'
outtiles_dir = '/media/ljp238/6tb/Joseph/DATASETS/OUT_TILES/1x1degree/'
tilenames = os.listdir(outtiles_dir)


if __name__ == '__main__':
    with ProcessPoolExecutor(50) as PPE:
        for i in range(len(tilenames)):
            #if i > 0: break

            tile_patches_path = os.path.join(outtiles_dir,tilenames[i],pathdir)
            print(tile_patches_path)
            fs = glob(f'{tile_patches_path}/*/*.tif')
            print(len(fs))
            #dirlistfiles(tile_patches_path, 'tif', True)
            for fi in fs: PPE.submit(correct_filenaming, fi)

print('Done')