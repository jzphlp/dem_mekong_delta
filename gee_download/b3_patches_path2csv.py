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
    for i in range(len(tilenames)):

        """ function start here to paralize"""

        tile_patches_path = os.path.join(outtiles_dir,tilenames[i],pathdir)
        print(tile_patches_path)

        fs = glob(f'{tile_patches_path}/*/*.tif')
        print(len(fs))

        zdif_files = files_filter_list(fs, 'zdif')
        lidar_files = files_filter_list(fs, 'LiDAR_MekongDelta_EGM08')

        txd_files = files_filter_list(fs, 'TDX_DEM_EGM08')
        cop_files = files_filter_list(fs, 'COP_DEM')
        merit_files = files_filter_list(fs, 'MERIT_EGM08')
        nasa_files = files_filter_list(fs, 'NASA_H_EGM08')
        aw3d_files = files_filter_list(fs, 'AW3D_H_EGM08')

        egm96_files = files_filter_list(fs, 'emg96')
        s1_files = files_filter_list(fs, 'S1')
        s2_files = files_filter_list(fs, 'S2')
        aspect_files = files_filter_list(fs, 'ASPECT')
        slope_files = files_filter_list(fs, 'SLOPE') 
        roughness_files = files_filter_list(fs, 'roughness') 
        tpi_files = files_filter_list(fs, 'TPI')
        tri_files = files_filter_list(fs, 'TRI')

        copwbm_files = files_filter_list(fs, 'COP_WBM')
        ethm_files = files_filter_list(fs, 'ETHm')
        eths_files = files_filter_list(fs, 'ETHs')
        fnf_files = files_filter_list(fs, 'TDX_FNF')
        esawc_files = files_filter_list(fs, 'WC')

        barea_files = files_filter_list(fs, 'BuildingArea')
        bfraction_files = files_filter_list(fs, 'BuildingFraction')
        bheight_files = files_filter_list(fs, 'BuildingHeight')
        bvolume_files = files_filter_list(fs, 'BuildingVolume')
        wsf_files = files_filter_list(fs, 'WSF') # fix the naming 
        wsf_files = [i for i in wsf_files if 'WSF3D' not in i]; print(len(wsf_files))


        ds = {}

        ds['zdif'] = zdif_files
        ds['lidar'] = lidar_files
        ds['tdemx'] = txd_files
        ds['cop'] = cop_files
        ds['nasa'] = nasa_files
        ds['aw3d']= aw3d_files
        ds['egm96'] = egm96_files
        ds['s1'] = s1_files
        ds['s2'] = s2_files
        ds['aspect'] = aspect_files
        ds['slope'] = slope_files
        ds['roughx'] = roughness_files
        ds['tpi'] = tpi_files
        ds['tri'] = tri_files
        ds['copwbm'] = copwbm_files
        ds['ethm'] = ethm_files
        ds['eths'] = eths_files
        ds['fnf'] = fnf_files
        ds['wcover']= esawc_files
        ds['barea'] = barea_files
        ds['bfraction'] = bfraction_files
        ds['bheight'] = bheight_files
        ds['bvolume'] = bvolume_files
        ds['wsf'] = wsf_files

        df = pd.DataFrame(ds)
        print(df.info())

        csv_tile_patches_path = os.path.join(tile_patches_path, f'{tilenames[i]}_patches_paths.csv')
        df.to_csv(csv_tile_patches_path, index=False)