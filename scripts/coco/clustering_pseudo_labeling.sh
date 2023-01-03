OUT_DIR=$1
MODEL_DIR=$2
FOLD=$3
NCLS=$4


echo 'output dir is ' $OUT_DIR
echo 'Load checkpoint from ' $MODEL_DIR
echo 'Run for ' $FOLD
echo 'Kmeans for ' $NCLS 'Clusters'

python base_class_remove.py --config configs/coco/base_train.yml --output-dir $OUT_DIR \
                            --fold $FOLD --ckpt $2/last_model.pth.tar --data-dir /ssd/yyzhao/data/coco

python generate_new_list.py --novel-dir $OUT_DIR --fold $FOLD --data-dir /ssd/yyzhao/data/coco --coco

python kmeans_novel.py --config configs/coco/kmeans_imgnet.yml --novel-dir $OUT_DIR \
                            --fold $FOLD --nclusters $NCLS --data-dir /ssd/yyzhao/data/coco

python mask_fuse.py --novel-dir $OUT_DIR --fold $FOLD --nclusters $NCLS --coco


