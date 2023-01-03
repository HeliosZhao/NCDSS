
python finetune_basic.py --config configs/coco/finetune_basic.yml \
--fold fold0 --output-dir output/coco_finetune_f0_basic --novel-dir output/coco_pseudos_fold0 --nclusters 40 

python finetune_basic.py --config configs/coco/finetune_basic.yml \
--fold fold1 --output-dir output/coco_finetune_f1_basic --novel-dir output/coco_pseudos_fold1 --nclusters 40 

python finetune_basic.py --config configs/coco/finetune_basic.yml \
--fold fold2 --output-dir output/coco_finetune_f2_basic --novel-dir output/coco_pseudos_fold2 --nclusters 40 

python finetune_basic.py --config configs/coco/finetune_basic.yml \
--fold fold3 --output-dir output/coco_finetune_f3_basic --novel-dir output/coco_pseudos_fold3 --nclusters 40 
