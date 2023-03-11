
from concurrent.futures import ProcessPoolExecutor
from geotile import GeoTile
#rom grid_raster_utils import *
#from FUtils import *
from glob import glob 
from osgeo import gdal, gdalconst

import os 
import time 
import rasterio


import geopandas as gpd
import pandas as pd 
import numpy as np 



def tileindex(tdir,gpkg): #pkg = tdir + '.gpkg'
    cmd = f'gdaltindex -t_srs EPSG:4326 -f GPKG {gpkg} {os.path.join(tdir)}/*.tif '
    os.system(cmd)

def retile(fi, tdir,C=True,ps=256):
    # it can take single or multi bands
    # Byte for cats: near or mode
    gpkg = tdir + '.gpkg'
    if C is True:
        dt='Float32';algo ='bilinear'
        cmd = f'gdal_retile.py -co compress=lzw -ot {dt} -ps {ps} {ps} -r {algo} -targetDir {tdir} {fi}'
    else:
        dt='Byte';algo ='near'
        cmd = f'gdal_retile.py -co compress=lzw -ot {dt} -ps {ps} {ps} -r {algo} -targetDir {tdir} {fi}'
    # open uw and see the code since it just python script  -v
    os.system(cmd)

    tileindex(tdir,gpkg)

def geotile_generate_tiles(tif_path, gpkg_path, dir_path):
    from geotile import GeoTile
    gt = GeoTile(tif_path)
    gt.generate_tiles(dir_path,tile_x=256, tile_y=256, stride_x=256, stride_y=256)
    gt.close()

    #gpkg = dir_path + '.gpkg'
    #gpkg = os.path.join(dir_path+'.gpkg')
    tileindex(dir_path,gpkg_path) #gpkg = tdir + '.gpkg'


def bounds2tilename(xmin,ymin):
    if ymin > 0 :
        ta = f'N{ymin}'
        if len(ta) != 3:
            ta = f'N0{ymin}'
    elif ymin < 0: 
        ta = f'S{abs(ymin)}'
    if xmin > 0 : 
            tb = f'E{xmin}'
    elif xmin < 0: 
        tb = f'W{xmin}'

    tile_name = f'{ta}_{tb}'
    tile_name = tile_name.replace('.','p')
    print(tile_name)


def dem_geoid_h2H(Hfile, hfile, Nfile):
    cmd = f'gdal_calc.py -A {hfile} -B {Nfile} --outfile={Hfile} --calc="A-B"'
    os.system(cmd)
    return Hfile 

def get_zdiff(zdif,dsm,dtm):
    cmd = f'gdal_calc.py -A {dsm} -B {dtm} --outfile={zdif} --calc="A-B"'
    os.system(cmd)
    return zdif 

def dem_geoid_H2H(Hyfile, Nyfile,Hxfile, Nxfile):
    cmd = f'gdal_calc.py -A {Hxfile} -B {Nxfile} -C {Nyfile} --outfile={Hyfile} --calc="A+B-C"'
    os.system(cmd)
    return Hyfile

def col2mat(col, patch_size=256):
    img = np.array(col).reshape((patch_size, patch_size))
    return img

def get_geotiff_nodatavalue(tif_path):
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    ds = None
    return nodata


xyres = 0.0001111111111111111164
def gdal_regrid(fi, fo,xmin,ymin, xmax, ymax,
                algo='cubicspline',datatype='Float32', res=xyres,t_epsg='EPSG:4326', overwrite=0):
    try:
        if overwrite == 0:
            #srcndv = str(get_geotiff_nodatavalue(fi)) -srcnodata {srcndv} 
           # dsnndv = -9999. -dstnodata {dsnndv}\

            cmd = f'gdalwarp -ot {datatype} -multi \
                -te {xmin} {ymin} {xmax} {ymax} \
                -r {algo}  -t_srs {t_epsg} -tr {res} {-res} -tap\
                -co compress=lzw -co num_threads=all_cpus \
                {fi} {fo}'
            # -t_srs {t_epsg} # -te_srs {t_epsg}\ -tr {res} {-res} \ #-tap with -tr xres yres
        os.system(cmd)
       # time.sleep(0.2)
       # gdal_edit_ndv(fo, dsnndv)
    except:
        pass
    ### write the other function when overwrite True :1 
    return fo 

def gdal_unsetndv(path):
    # unsetnodata: remove exisiting nodata 
    # -a_nodata : assing a specific nodata value
   # cmd = f'gdal_edit.py -unsetnodata -a_nodata {ndv} {path}'
    cmd = f'gdal_edit.py -unsetnodata {path}'
    os.system(cmd)


def gdal_edit_ndv(path, ndv):
    # unsetnodata: remove exisiting nodata 
    # -a_nodata : assing a specific nodata value
    cmd = f'gdal_edit.py -a_nodata {ndv} {path}'
    os.system(cmd)

def get_raster_info(tif_path):
    """Function to read a GeoTIFF raster (given its path)
    and return its projection, resolution, bounding box,
    width & height, as a list of separate variables."""
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    # Extract projection & geotransform from input dataset
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    xres = geotrans[1]
    yres = geotrans[5]  # Note: identical to x-resolution, but negative
    w = ds.RasterXSize
    h = ds.RasterYSize
    # Get bounding box of input dataset
    xmin = geotrans[0]
    ymax = geotrans[3]
    xmax = xmin + (geotrans[1] * w)
    ymin = ymax + (geotrans[5] * h)
    # Close access to GeoTIFF file
    ds = None
    # Return all results
    return proj, xres, yres,xmin, xmax, ymin, ymax, w, h

### bounded rasters together with tdemx clean up 

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




def preprocessing1(
    xmin, ymin, xmax, ymax,outdir_tile,tile_name,
    REGION, lgeoid_file, lidar_file, egm96_file,egm08_file,
    aw3dh_file,aw3dH_file,merit_file,nasa_file,copdem_file,
    copwbm_file,ethm_file,eths_file,esawc_file,wsf_file,
    tdxcom_file,tdxcov_file,tdxfnf_file,tdxwam_file,
    tdxdem_file,sent1_file,sent2_file
):
    
    
    ds = {}
    lgeoid_tile = os.path.join(outdir_tile, tile_name +'_Lgeoid_'+os.path.basename(lgeoid_file))
    if os.path.isfile(lgeoid_tile): print(f'File already created {lgeoid_tile}')
    else: lgeoid_tile = gdal_regrid(lgeoid_file, lgeoid_tile,xmin, ymin, xmax, ymax)
    ds['lgeoid'] = lgeoid_tile

    lidar_tile = os.path.join(outdir_tile, tile_name +'_LiDAR_'+os.path.basename(lidar_file))
    if os.path.isfile(lidar_tile): print(f'File already created {lidar_tile}')
    else: lidar_tile = gdal_regrid(lidar_file, lidar_tile,xmin, ymin, xmax, ymax)
    ds['lidar'] = lidar_tile

    tandemx_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(tdemx_file)).replace('.vrt','.tif')
    if os.path.isfile(tandemx_tile): print(f'File already created {tandemx_tile}')
    else: tandemx_tile = gdal_regrid(tdemx_file, tandemx_tile, xmin, ymin, xmax, ymax)
    ds['tdemx'] = tandemx_tile

    egm08_tile = os.path.join(outdir_tile, tile_name +'_'+ 'emg08.tif')
    if os.path.isfile(egm08_tile): print(f'File already created {egm08_tile}')
    else: egm08_tile = gdal_regrid(egm2008_file, egm08_tile, xmin, ymin, xmax, ymax)
    ds['egm08'] = egm08_tile

    egm96_tile = os.path.join(outdir_tile, tile_name +'_'+ 'emg96.tif')
    if os.path.isfile(egm96_tile): print(f'File already created {egm96_tile}')
    else: egm96_tile = gdal_regrid(egm1996_file, egm96_tile,xmin, ymin, xmax, ymax)
    ds['egm96'] = egm96_tile

    nasadem_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(nasa_file)).replace('.vrt','.tif')
    if os.path.isfile(nasadem_tile): print(f'File already created {nasadem_tile}')
    else: nasadem_tile = gdal_regrid(nasa_file, nasadem_tile, xmin, ymin, xmax, ymax)
    ds['nasa'] = nasadem_tile

    aw3d_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(aw3d_file)).replace('.vrt','.tif')
    if os.path.isfile(aw3d_tile): print(f'File already created {aw3d_tile}')
    else: aw3d_tile = gdal_regrid(aw3d_file, aw3d_tile, xmin, ymin, xmax, ymax)
    ds['aw3d'] = aw3d_tile

    cop_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(cop_file)).replace('.vrt','.tif')
    if os.path.isfile(cop_tile): print(f'File already created {cop_tile}')
    else: gdal_regrid(cop_file, cop_tile, xmin, ymin, xmax, ymax)
    ds['cop'] = cop_tile

    merit_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(merit_file)).replace('.vrt','.tif')
    if os.path.isfile(merit_tile): print(f'File already created {merit_tile}')
    else: gdal_regrid(merit_file, merit_tile, xmin, ymin, xmax, ymax)
    ds['merit'] = merit_tile

    chmm_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(chmm_file)).replace('.vrt','.tif')
    if os.path.isfile(chmm_tile): print(f'File already created {chmm_tile}')
    else: chmm_tile = gdal_regrid(chmm_file, chmm_tile, xmin, ymin, xmax, ymax,algo_c,dtype_c)
    ds['chm_map'] = chmm_tile

    chms_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(chms_file)).replace('.vrt','.tif')
    if os.path.isfile(chms_tile): print(f'File already created {chms_tile}')
    else: chms_tile = gdal_regrid(chms_file, chms_tile, xmin, ymin, xmax, ymax,algo_c,dtype_c)
    ds['chm_std'] = chms_tile

    wsf2019_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(wsf2019_file)).replace('.vrt','.tif')
    if os.path.isfile(wsf2019_tile): print(f'File already created {wsf2019_tile}')
    else: wsf2019_tile = gdal_regrid(wsf2019_file, wsf2019_tile, xmin, ymin, xmax, ymax,algo_c,dtype_c)
    ds['wsf2019'] = wsf2019_tile

    esri_lc20_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(esri_lc20_file)).replace('.vrt','.tif')
    if os.path.isfile(esri_lc20_tile): print(f'File already created {esri_lc20_tile}')
    else: esri_lc20_tile = gdal_regrid(esri_lc20_file, esri_lc20_tile, xmin, ymin, xmax, ymax,algo_c,dtype_c)
    ds['esri_lc'] = esri_lc20_tile
    
    esa_wc21_tile = os.path.join(outdir_tile, tile_name +'_'+os.path.basename(esa_wc21_file)).replace('.vrt','.tif')
    if os.path.isfile(esa_wc21_tile): print(f'File already created {esa_wc21_tile}')
    else: esa_wc21_tile = gdal_regrid(esa_wc21_file, esa_wc21_tile, xmin, ymin, xmax, ymax,algo_c,dtype_c)
    ds['esa_wc'] = esa_wc21_tile

    # vertical datum transformations tdemx,lidar, [zdiff],aw3d,nasa,merit 


    ########################################################################################################################
    # tdemx h2H #### compression and other options
    tandemx_tile_egm08 = tandemx_tile.replace('.tif', '_EGM08.tif')
    if os.path.isfile(tandemx_tile_egm08): print(f'File already created {tandemx_tile_egm08}')
    else: tandemx_tile_egm08 = dem_geoid_h2H(tandemx_tile_egm08, tandemx_tile, egm08_tile)
    ds['tdemx_egm08'] = tandemx_tile_egm08

    lidar_tile_egm08 = lidar_tile.replace('.vrt', '_EGM08.tif')
    if os.path.isfile(lidar_tile_egm08): print(f'File already created {lidar_tile_egm08}')
    else: lidar_tile_egm08 = dem_geoid_H2H(lidar_tile_egm08, egm08_tile,lidar_tile, lgeoid_tile)
    #gdal_regrid(lidar_file, lidar_tile,xmin, ymin, xmax, ymax)
    ds['lidar_egm08'] = lidar_tile_egm08

    zdif_tile = os.path.join(outdir_tile, tile_name +'_'+'zdif.tif')
    if os.path.isfile(zdif_tile): print(f'File already created {zdif_tile}')
    else: zdif_tile = get_zdiff(zdif_tile, tandemx_tile_egm08,lidar_tile_egm08)
    ds['zdif'] = zdif_tile

    nasadem_tile_egm08 = nasadem_tile.replace('.tif', '_EGM08.tif')
    if os.path.isfile(nasadem_tile_egm08): print(f'File already created {nasadem_tile_egm08}')
    else: nasadem_tile = dem_geoid_H2H(nasadem_tile_egm08, egm08_tile,nasadem_tile, egm96_tile)
    #gdal_regrid(nasa_file, nasadem_tile, xmin, ymin, xmax, ymax)
    ds['nasa_egm08'] = nasadem_tile_egm08

    aw3d_tile_egm08 = aw3d_tile.replace('.tif', '_EGM08.tif')
    if os.path.isfile(aw3d_tile_egm08): print(f'File already created {aw3d_tile_egm08}')
    else: aw3d_tile_egm08 = dem_geoid_H2H(aw3d_tile_egm08, egm08_tile,aw3d_tile, egm96_tile)
    #gdal_regrid(aw3d_file, aw3d_tile, xmin, ymin, xmax, ymax)
    ds['aw3d_egm08'] = aw3d_tile_egm08

    merit_tile_egm08  = merit_tile.replace('.tif', '_EGM08.tif')
    if os.path.isfile(merit_tile_egm08): print(f'File already created {merit_tile_egm08}')
    else: merit_tile_egm08 = dem_geoid_H2H(merit_tile_egm08, egm08_tile,merit_tile, egm96_tile)
    #gdal_regrid(merit_file, merit_tile, xmin, ymin, xmax, ymax)
    ds['merit_egm08'] = merit_tile_egm08


    csv_out = os.path.join(outdir_tile,f'{REGION}_{tile_name}.csv')
    if os.path.isfile(csv_out):
        print(f'File already exist {csv_out}')
    else:
        print(f'creating a new {csv_out}')
        tnames  = list(ds.keys())
        tpaths  = list(ds.values())
        df = pd.DataFrame(tpaths, columns=['path'])
        df['name'] = tnames
        df.to_csv(csv_out, index=True)
