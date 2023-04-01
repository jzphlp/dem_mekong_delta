import os 
import time 

# load all b_ script.py  into a list 
b0 = '/home/ljp238/Documents/phdprojects/dem_mekong_delta_pkg/dem_mekong_delta/gee_download/b0_tiling.py'
b1 = '/home/ljp238/Documents/phdprojects/dem_mekong_delta_pkg/dem_mekong_delta/gee_download/b1_patching.py'
b2 = ''
b_scripts_list = [b0,b1,b2]

for i in b_scripts_list:
    try:
        os.system(i)
        time.sleep(0.2)
    except:
        pass 
