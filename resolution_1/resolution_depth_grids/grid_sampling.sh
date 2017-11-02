#!/bin/bash

### Script to sample and extract depth grids along the survey coords ###
rm *extraction*
rm *grd*

# This script grids a horizon file, and then samples it at every shotpoint from the survey_coords file, and writes it to a text file.
for i in `ls *dat`
do

out_grid_name=`echo $i | awk 'BEGIN{FS="."}{print $1 ".grd"}'`
out_file_name=`echo $i | awk 'BEGIN{FS="."}{print $1 "_extraction.dat"}'`

surface $i -G$out_grid_name -I50 `gmtinfo $i -I50` -V
grdtrack -G$out_grid_name survey_coords.xy > $out_file_name

done
