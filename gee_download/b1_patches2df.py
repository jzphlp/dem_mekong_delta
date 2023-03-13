from UtilsFunc import dirlistfiles,filterlist,files_filter_list
import os 
import pandas as pd

pathdir = 'patch256x256'
outtiles_dir = '/media/ljp238/6tb/Joseph/DATASETS/OUT_TILES/1x1degree/'
tilenames = os.listdir(outtiles_dir)


vnames = ['zdif','lidar','tdx','cop','merit','nasa','aw3d','egm96', 'ethm','eths',
          'fnf','esawc','wsf', 's1vv','s1vh','s1a','s2r','s2g','s2b']

namefiles = ['zdif','lidar','tdx','cop','merit','nasa','aw3d','egm96', 'ethm','eths',
             'fnf','esawc','wsf', 's1','s2']
### the constants like in probHAND

if __name__ == '__main__':

    for i in range(len(tilenames)):

        tile_patches_path = os.path.join(outtiles_dir,tilenames[i],pathdir)
        print(tile_patches_path)
        fs = dirlistfiles(tile_patches_path, 'tif', True)
        for fi in fs:
            fo = fi.replace('..tif','.tif')
            os.rename(fi, fo)

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

        ethm_files = files_filter_list(fs, 'ETHm')
        eths_files = files_filter_list(fs, 'ETHs')
        fnf_files = files_filter_list(fs, 'TDX_FNF')
        esawc_files = files_filter_list(fs, 'WC')
        wsf_files = files_filter_list(fs, 'WSF')

        df = pd.DataFrame(list(zip(zdif_files,lidar_files,txd_files,
                            cop_files,merit_files,nasa_files,
                            aw3d_files,egm96_files,ethm_files,
                            eths_files,fnf_files,esawc_files,
                            wsf_files,s1_files,s2_files)),
                    columns=namefiles)
        
        csv_tile_patches_path = os.path.join(tile_patches_path, f'{tilenames[i]}_patches_paths.csv')
        df.to_csv(csv_tile_patches_path, index=False)




