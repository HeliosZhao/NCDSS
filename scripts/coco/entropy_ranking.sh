
## obtain the entropy ranking after basic framework
python entropy_ranking.py --config configs/coco/finetune_basic.yml \
--fold fold0 --output-dir output/entropy --novel-dir output/coco_pseudos_fold0 --nclusters 40 \
--ckpt output/coco_finetune_f0_basic/best_model.pth.tar --split-dir entropy_ranking23 --ent 0.67

## obtain the entropy ranking after basic framework
python entropy_ranking.py --config configs/coco/finetune_basic.yml \
--fold fold1 --output-dir output/entropy --novel-dir output/coco_pseudos_fold1 --nclusters 40 \
--ckpt output/coco_finetune_f1_basic/best_model.pth.tar --split-dir entropy_ranking23 --ent 0.67

## obtain the entropy ranking after basic framework
python entropy_ranking.py --config configs/coco/finetune_basic.yml \
--fold fold2 --output-dir output/entropy --novel-dir output/coco_pseudos_fold2 --nclusters 40 \
--ckpt output/coco_finetune_f2_basic/best_model.pth.tar --split-dir entropy_ranking23 --ent 0.67

## obtain the entropy ranking after basic framework
python entropy_ranking.py --config configs/coco/finetune_basic.yml \
--fold fold3 --output-dir output/entropy --novel-dir output/coco_pseudos_fold3 --nclusters 40 \
--ckpt output/coco_finetune_f3_basic/best_model.pth.tar --split-dir entropy_ranking23 --ent 0.67
