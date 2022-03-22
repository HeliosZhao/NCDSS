
python finetune_basic.py --config configs/finetune_basic.yml \
--fold fold0 --output-dir output/finetune_f0_basic --novel-dir output/pseudos_fold0 --nclusters 10 

python finetune_basic.py --config configs/finetune_basic.yml \
--fold fold1 --output-dir output/finetune_f1_basic --novel-dir output/pseudos_fold1 --nclusters 10 

python finetune_basic.py --config configs/finetune_basic.yml \
--fold fold2 --output-dir output/finetune_f2_basic --novel-dir output/pseudos_fold2 --nclusters 10 

python finetune_basic.py --config configs/finetune_basic.yml \
--fold fold3 --output-dir output/finetune_f3_basic --novel-dir output/pseudos_fold3 --nclusters 10 
