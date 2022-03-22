## EUMS framework

python finetune_eums.py --config configs/finetune_eums.yml --fold fold0 --output-dir output/finetune_f0_eums --novel-dir output/pseudos_fold0 --nclusters 10 --split-dir entropy_ranking23

python finetune_eums.py --config configs/finetune_eums.yml --fold fold1 --output-dir output/finetune_f1_eums --novel-dir output/pseudos_fold1 --nclusters 10 --split-dir entropy_ranking23

python finetune_eums.py --config configs/finetune_eums.yml --fold fold2 --output-dir output/finetune_f2_eums --novel-dir output/pseudos_fold2 --nclusters 10 --split-dir entropy_ranking23

python finetune_eums.py --config configs/finetune_eums.yml --fold fold3 --output-dir output/finetune_f3_eums --novel-dir output/pseudos_fold3 --nclusters 10 --split-dir entropy_ranking23


