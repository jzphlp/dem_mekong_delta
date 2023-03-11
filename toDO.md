
# TODO 
[x] add edit ndv 
[x] test the code >> TDM DEM null editing did not work well : 
[]  find orginal null values for TDMX and Lidar and edit null with 

next step will help 
### pipeline 2 {add to the code for tiling}:
###### deal with tdx null values - values in tdemx clean up 
###### deal with categorical data : fill classes like you'd for tabluar - eg. wsf as binary  
 

 start a new branch called "wx" to work on changes from here 
 NB: you need to clean up the tdmex before you can do the zdiff#  it relies on TDXDEM

 if tdxcleanup.py is done before the tiling.py : then we can create less layers
 run through subprocess.run() or os.system() like for the lisflood:
    tdxcleanup.py
    tiling.py
    patching.py
    1D_desing_experiment.py: create 1DF by patches with ID, sampling for spatial cross validation, train, test, valid and data scaling
    1D_modllling_catboost.py [baseline] : catboost with many resampling techniques
    1D_inference.py : prediciton metrics and viz
    2D_modelling: DL4earth
    2D_psols_ddr.py
    2D_unet_geoww.py : 
    2D_bioms.py : 
    mlm_cgan.py
    deepbed.py
    cgan_terrain.py
    superres.py
    
    [i]2D_desing_experiment.py: npz label and features to feed tkf unert, biomaster like