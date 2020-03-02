# mpirun --tag-output ./setup.sh 
python train.py \
-data_path /nobackup/users/haft/OCTseg/ \
-exp_def $1 \
-nLayer 3 \
-nFeature 8 \
-lr 1e-4 \
-nBatch 200  \
-nEpoch 1000 \
-saveEpoch 100 \
-loss_w $2 

# mpirun --tag-output ./reset.sh
 
