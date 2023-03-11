
from UtilsTiling import * 
import multiprocessing
import os 
import time 



roi_path = '/media/ljp238/6tb/Joseph/DATASETS/ROI_FILES'

aw3dh_file  = f'{roi_path}/AW3D_h.vrt'
aw3dH_file  = f'{roi_path}/AW3D_H.vrt'
merit_file = f'{roi_path}/MERIT.vrt'
nasa_file = f'{roi_path}/NASA_H.vrt'

lidar_file = f'{roi_path}/LiDAR_MekongDelta.vrt' 
sent1_file = f'{roi_path}/S1.vrt'
sent2_file = f'{roi_path}/S2.vrt' 

copwbm_file = f'{roi_path}/COP_WBM.vrt' 
copdem_file = f'{roi_path}/COP_DEM.vrt'

ethm_file = f'{roi_path}/ETHm.vrt'
eths_file = f'{roi_path}/ETHs.vrt'
esawc_file = f'{roi_path}/WC.vrt'
wsf_file = f'{roi_path}/WSF.vrt'

tdxcom_file = f'{roi_path}/TDX_COM.vrt'
tdxcov_file = f'{roi_path}/TDX_COV.vrt'
tdxdem_file = f'{roi_path}/TDX_DEM.vrt'
tdxfnf_file = f'{roi_path}/TDX_FNF.vrt'
tdxwam_file = f'{roi_path}/TDX_WAM.vrt'

geoids_path = '/media/ljp238/6tb/Joseph/DATASETS/GEOIDS/geoid_grids'
lgeoid_file = f'{geoids_path}/GEOID_FFT_20190703_Vietnam.tif'
egm08_file =  f'{geoids_path}/egm2008-1.tif'
egm96_file =  f'{geoids_path}/us_nga_egm96_15.tif'


REGION = 'MekongDelta'
VGRIDS = '/media/ljp238/6tb/Joseph/DATASETS/GRIDS/' # /media/ljp238/6tb/Joseph/DATASETS/VECTOR_GRIDS/
WDIR= '/media/ljp238/6tb/Joseph/DATASETS/OUT_TILES/'
STEP = 1
CPUS = int(os.cpu_count() * 0.9)
xstep = str(STEP).replace('.', 'p')
wdir = os.path.join(WDIR, f'{xstep}x{xstep}degree')
os.makedirs(wdir, exist_ok=True)


base_path = '/media/ljp238/6tb/Joseph/DATASETS/DEMs/TANDEMX/' #########
base_files = sorted(glob(base_path+'*.tif')); print(len(base_files))

if __name__ == '__main__':
    ti = time.perf_counter()

    pool = multiprocessing.Pool()
    result_async = []
    for i in range(len(base_files)):
       # if i >=2 : break
        tif_path = base_files[i]
        proj, resx, resy, minx, maxx, miny, maxy,width,height = get_raster_info(tif_path)
        nodata = get_geotiff_nodatavalue(tif_path)
        
        print(minx, miny,maxx,maxy)
        xmin,ymin, xmax, ymax = int(round(minx)), int(round(miny)), int(round(maxx)), int(round(maxy))
        print(xmin,ymin, xmax, ymax)
        tile_name = bounds2tilename(xmin,ymin)
        print(tile_name,'tile_name')
        outdir_tile = os.path.join(wdir, tile_name)
        os.makedirs(outdir_tile, exist_ok=True)

        result_async.append(pool.apply_async(tiling_pipeline,  args=(xmin, ymin, xmax, ymax,outdir_tile,tile_name,
                                                            REGION, lgeoid_file, lidar_file, egm96_file,egm08_file,
                                                            aw3dh_file,aw3dH_file,merit_file,nasa_file,copdem_file,
                                                            copwbm_file,ethm_file,eths_file,esawc_file,wsf_file,
                                                            tdxcom_file,tdxcov_file,tdxfnf_file,tdxwam_file,
                                                            tdxdem_file,sent1_file,sent2_file,)))
    
    results = [r.get() for r in result_async]
    print("Output: {}".format(results))
    tf = time.perf_counter() - ti 
    print(f'run.time {tf/60} min(s)')