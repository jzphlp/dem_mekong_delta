from UtilsFunc import *
import os
import time 
import pandas as pd


outdir = '/media/ljp238/6tb/Joseph/DATASETS/COPDEM/'
vois = ['WBM', 'EDM', 'FLM']

tile_names = os.listdir(outdir)
tile_names = [i for i in tile_names if not 'WBM' in i and not 'OUT' in i]
print(tile_names)

if __name__ == '__main__':
    ti = time.perf_counter()
   
    vois_paths = []
    for voi in vois:
        #print(voi)
        for i in range(len(tile_names)):
            tile_path = os.path.join(outdir, tile_names[i])
            tile_path_var = os.path.join(tile_path, voi)
            vrt_voi_path = os.path.join(tile_path, f'{tile_names[i]}_{voi}.vrt')
            buildVRT_bypath(vrt_voi_path, tile_path_var)
            #print(vrt_voi_path)
            vois_paths.append({'TileName': tile_names[i], 'Variable': voi, 'Path': vrt_voi_path})

    merge_out = os.path.join(outdir, 'OUT')
    os.makedirs(merge_out, exist_ok=True)
    df = pd.DataFrame(vois_paths)
    df.to_csv(os.path.join(merge_out,'vrt_paths.csv'), index=False)

    for voi in vois:
        print(voi)
        voi_tiles = df[df.Variable == voi].Path.tolist()
        #print(voi_tiles)#
        name = voi_tiles[0].split('/')[-2]
        txti = os.path.join(merge_out,f'{voi}.txt')
        vrti = txti.replace('.txt','.vrt')

        writepath2txt(txti, voi_tiles)
        #time.sleep(0.1)
        buildVRT_bytxt(txti, vrti)

    tf = time.perf_counter() - ti 
    print(f'run.tim {tf/60} min(s)')