## Evaluate EUMS
python eval.py --config configs/finetune_eums.yml --fold fold0 --output-dir output/eval_f0 --nclusters 10 --ckpt eums_model/eums_f0.pth.tar

python eval.py --config configs/finetune_eums.yml --fold fold1 --output-dir output/eval_f1 --nclusters 10 --ckpt eums_model/eums_f1.pth.tar

python eval.py --config configs/finetune_eums.yml --fold fold2 --output-dir output/eval_f2 --nclusters 10 --ckpt eums_model/eums_f2.pth.tar

python eval.py --config configs/finetune_eums.yml --fold fold3 --output-dir output/eval_f3 --nclusters 10 --ckpt eums_model/eums_f3.pth.tar
