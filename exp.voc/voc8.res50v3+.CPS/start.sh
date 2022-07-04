
#export volna="/media/data/jinchao/TorchSemiSeg/"
export volna="../../"
export NGPUS=4
export OUTPUT_PATH=$volna"/results"
export snapshot_dir=$OUTPUT_PATH

export batch_size=8
export learning_rate=0.0025
export snapshot_iter=1

export labeled_ratio=16
export nepochs=32

#TIME=$(date +"TorchSemiSeg_%Y-%m-%d-%H-%M-%S")
#LOG=$TIME.txt

## following is the command for train and eval
#export NGPUS=3
#export batch_size=6
#echo $NGPUS
#export CUDA_VISIBLE_DEVICES=0,1,2
##python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py >> $LOG
#python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
#2>&1 | tee $LOG
#
#export TARGET_DEVICE=$[$NGPUS-1]
#echo $NGPUS
#echo $TARGET_DEVICE
##python eval.py -e 20-34 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results
##python eval.py -e 20-34 -d 0-3 --save_path $OUTPUT_PATH/results
#python eval.py -e 0-1 -d 0 --save_path $OUTPUT_PATH/
#2>&1 | tee $LOG


# following is the command for debug
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
export batch_size=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1
