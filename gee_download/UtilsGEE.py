import ee 
import os 
import geemap
import geopandas as gpd 
import numpy as np
import time
from glob import glob
from geotile import GeoTile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def tileindex(tdir,gpkg): #pkg = tdir + '.gpkg'
    cmd = f'gdaltindex -t_srs EPSG:4326 -f GPKG {gpkg} {tdir}/*.tif '
    os.system(cmd)

def geotile_generate_tiles(tif_path, gpkg_path, dir_path):
    from geotile import GeoTile
    
    if not os.path.isfile(gpkg_path):
        gt = GeoTile(tif_path)
        gt.generate_tiles(dir_path,tile_x=256, tile_y=256, stride_x=256, stride_y=256)
        gt.close()

        #gpkg = dir_path + '.gpkg'
        #gpkg = os.path.join(dir_path+'.gpkg')
        tileindex(dir_path,gpkg_path) #gpkg = tdir + '.gpkg'
    else: print(f'gpkg file already exists {gpkg_path}')

def ee_clip_mosaic_roi(dobject,roi):
    mosaic = dobject.mosaic()
    mosaiclip = mosaic.clip(roi)
    return mosaiclip


def getDEM_files(roi):
    glo30  = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(roi)
  #  dem = glo30.select('DEM')
    #edm = glo30.select('EDM')
   # flm = glo30.select('FLM')
   # hem = glo30.select('HEM')
   # wbm = glo30.select('WBM')

    wbm = ee_clip_mosaic_roi(glo30.select('WBM'),roi)
    hem = ee_clip_mosaic_roi(glo30.select('HEM'),roi)
    flm = ee_clip_mosaic_roi(glo30.select('FLM'),roi)
    edm = ee_clip_mosaic_roi(glo30.select('EDM'),roi)
    dem = ee_clip_mosaic_roi(glo30.select('DEM'),roi)

    
    return dem,edm,flm,hem,wbm

def get_ee_geometry(i, g, name):
    ig = g.iloc[i:i+1,]
    bBox = [float(ig.minx), float(ig.miny), float(ig.maxx), float(ig.maxy)]
    fname = (os.path.basename(ig.location.values[0])).replace('..tif', f'_{name}.tif')
    region = ee.Geometry.Rectangle(bBox)
    return region, fname

def gee_download_geemap(image,outpath, scale):
    print(outpath)
   # image = ee.Image(image)
    if os.path.isfile(outpath):
        print('Already downloaded') 
    else:
        geemap.ee_export_image(image, outpath, scale=scale)




