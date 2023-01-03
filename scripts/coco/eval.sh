## Evaluate EUMS
python eval.py --config configs/coco/finetune_eums.yml --fold fold0 --output-dir output/coco_eval_f0 --nclusters 40 --ckpt coco_eums_model/eums_f0.pth.tar

python eval.py --config configs/coco/finetune_eums.yml --fold fold1 --output-dir output/coco_eval_f1 --nclusters 40 --ckpt coco_eums_model/eums_f1.pth.tar

python eval.py --config configs/coco/finetune_eums.yml --fold fold2 --output-dir output/coco_eval_f2 --nclusters 40 --ckpt coco_eums_model/eums_f2.pth.tar

python eval.py --config configs/coco/finetune_eums.yml --fold fold3 --output-dir output/coco_eval_f3 --nclusters 40 --ckpt coco_eums_model/eums_f3.pth.tar
