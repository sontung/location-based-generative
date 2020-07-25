python train_uc.py --dir "/scratch/mlr/nguyensg/pbw/blocks-5-3" --eval_dir "/scratch/mlr/nguyensg/pbw/blocks-6-3" --nb_samples 5000 --device 5 --save_data 1 --epc 20
python train_uc_sim.py --dir "/scratch/mlr/nguyensg/pbw/5objs_seg" --eval_dir "/scratch/mlr/nguyensg/pbw/6objs_seg" --nb_samples 5000 --device 5 --epc 20 --save_data 1
python train_uc.py --dir "/scratch/mlr/nguyensg/pbw/blocks-5-3" --eval_dir "/scratch/mlr/nguyensg/pbw/blocks-6-3" --nb_samples -1 --device 5 --save_data 0 --epc 20
python train_uc_sim.py --dir "/scratch/mlr/nguyensg/pbw/5objs_seg" --eval_dir "/scratch/mlr/nguyensg/pbw/6objs_seg" --nb_samples -1 --device 5 --epc 20 --save_data 0
