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


def get_S1mosaic(aoi,pol='VV', opass='ASCENDING',idate='2019-01-01',fdate='2022-12-01'):
    sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filter(ee.Filter.eq('instrumentMode','IW')) \
    .filterDate(idate,fdate).filter(ee.Filter.listContains('transmitterReceiverPolarisation', pol)) \
    .filter(ee.Filter.eq('orbitProperties_pass',opass)) \
    .filter(ee.Filter.eq('resolution_meters',10)) \
    .filterBounds(aoi)\
    # play with params and see what kind of data you  can get
    s1img = ee.Image(sentinel1.mosaic().clip(aoi))
    
    return s1img


def gee_download_geemap(image,outpath, scale):
    print(outpath)
    if os.path.isfile(outpath):
        print('Already downloaded') 
    else:
        geemap.ee_export_image(image, outpath, scale=scale)

def getS1patch(i,g,pol,name,S1tile_path,scale):
    #if not os.path.isfile
    region, fname = get_ee_geometry(i, g,name)
    s1img = get_S1mosaic(region, pol)
    outpath = os.path.join(S1tile_path, fname)
    gee_download_geemap(s1img,outpath, scale)
    time.sleep(0.5)



def maskClouds(image):
    qa = image.select('QA60')
    cloudBitMask = int(2**10)
    cirrusBitMask = int(2**11)
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)


def getS2_RGBmedian(region,CLOUD_FILTER=10):
    s2coll = ee.ImageCollection('COPERNICUS/S2_SR') \
             .filterBounds(region) \
             .filterDate('2021', '2022') \
             .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
    

    sentinel2_masked = s2coll.map(maskClouds)
    rgb_bands = ['B4', 'B3', 'B2']
    rgb = sentinel2_masked.select(rgb_bands).median().clip(region)
    return rgb 

def getS2RGBpatch(i,g,name,S2tile_path,scale):
    region, fname = get_ee_geometry(i, g,name)
    rgb = getS2_RGBmedian(region)
    outpath = os.path.join(S2tile_path, fname)
    gee_download_geemap(rgb,outpath, scale)
    time.sleep(0.5)
