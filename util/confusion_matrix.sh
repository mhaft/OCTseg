# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

# running confusion matrix for all saved epochs

for i in {100..1000..100}
do
	echo '*******************   Model '$1'   Epoch   '$i'   ***********************'
	python confusion_matrix.py -epoch $i -exp_def $1 ;
done
