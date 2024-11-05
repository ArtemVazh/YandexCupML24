CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=configs/config.yaml python main.py train.max_size=-1 test.max_size=-1 train.label_smoothing=0.0 +train.mixup=True +train.conformer=False +train.model_name=chunk_opt model.num_layers=4
wait
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=configs/config.yaml python main.py train.max_size=-1 test.max_size=-1 train.label_smoothing=0.1 +train.mixup=True +train.conformer=False +train.model_name=chunk_opt model.num_layers=4
wait
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=configs/config.yaml python main.py train.max_size=-1 test.max_size=-1 +train.conformer=False +train.model_name=chunk_opt_v5 +train.mixup=True train.label_smoothing=0.1 model.num_layers=4 output_flag=-chunk_mixup_ls_overall
wait
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=configs/config.yaml python main.py train.max_size=-1 test.max_size=-1 +train.conformer=False +train.model_name=chunk_opt_v5 +train.mixup=True train.label_smoothing=0.1 model.num_layers=4 output_flag=-chunk_mixup_ls_overall_seed777 seed=777
wait
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=configs/config.yaml python main.py train.max_size=-1 test.max_size=-1 +train.conformer=False +train.model_name=chunk_opt_v5 +train.mixup=True train.label_smoothing=0.1 model.num_layers=4 output_flag=-chunk_mixup_ls_overall_seed123456789 seed=123456789