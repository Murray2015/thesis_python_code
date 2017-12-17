#!/bin/bash

#gmtset PS_MEDIA A2
myfile="/home/mxh909/Documents/random_work/resolution-1-magee/Resolution-1-well/x_y_line_trace_depth_amplitude/Sea_floor.dat"

## extract shotpoints
# file is currently in epsg 2193
awk 'BEGIN{FS=","}{print $1, $2}' $myfile  | cs2cs +init=epsg:2193 +to +proj=latlong +datum=WGS84 -f %.12f > coords.txt

location="-R165/179/-48/-33"
prj="-JM6i"
filename="res_map.ps"

makecpt -Cetopo1 -T-11/8.55/0.01 -Z -D > res.cpt
grdmath /home/mxh909/Documents/global_data/ETOPO1_Bed_g_gmt4.grd 1000 DIV = etopo1_km.nc
grdimage etopo1_km.nc $location $prj -Cres.cpt -K -P > $filename
psxy coords.txt $location $prj -K -O -W1,red >> $filename
psscale -D6.75i/4.4i/5i/0.5i -Cetopo1 -O -K -Ba1t1g1 >> $filename
pscoast $location $prj -Wblack -B2 -P -Dh -I0 -O >> $filename

echo mog
mogrify -trim -bordercolor white -border 30x30 -quality 100 -density 300 -format jpg $filename
eog *jpg
