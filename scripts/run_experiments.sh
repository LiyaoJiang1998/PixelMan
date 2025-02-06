# Ablation Experiments
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo ours --use_gsn 0 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 1 --steps 16
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo ours --use_gsn 1 --inversion_free 0 --sa_masking_ipt 1 --use_copy_paste 1 --steps 16
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 0 --use_copy_paste 1 --steps 16
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 0 --steps 16

# Comparisons

# Experiments: run at 16 steps
# Evaluation DragonDiffusion on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo dragon --use_prompt --steps 16
python run_move_static.py --which_dataset ReS --which_img all --edit_algo dragon --use_prompt --steps 16
# Evaluation DiffEditor on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo diffeditor --use_prompt --steps 16
python run_move_static.py --which_dataset ReS --which_img all --edit_algo diffeditor --use_prompt --steps 16
# Evaluation SelfGuidance on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo sg --use_prompt --steps 16
python run_move_static.py --which_dataset ReS --which_img all --edit_algo sg --use_prompt --steps 16
# Evaluation Ours on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 1 --steps 16
python run_move_static.py --which_dataset ReS --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 1 --steps 16

# Experiments: run at 50 steps
# Evaluation DragonDiffusion on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo dragon --use_prompt --steps 50
python run_move_static.py --which_dataset ReS --which_img all --edit_algo dragon --use_prompt --steps 50
# Evaluation DiffEditor on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo diffeditor --use_prompt --steps 50
python run_move_static.py --which_dataset ReS --which_img all --edit_algo diffeditor --use_prompt --steps 50
# Evaluation SelfGuidance on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo sg --use_prompt --steps 50
python run_move_static.py --which_dataset ReS --which_img all --edit_algo sg --use_prompt --steps 50
# Evaluation Ours on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 1 --steps 50
python run_move_static.py --which_dataset ReS --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 1 --steps 50

# Experiments: run at 8 steps
# Evaluation DragonDiffusion on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo dragon --use_prompt --steps 8
python run_move_static.py --which_dataset ReS --which_img all --edit_algo dragon --use_prompt --steps 8
# Evaluation DiffEditor on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo diffeditor --use_prompt --steps 8
python run_move_static.py --which_dataset ReS --which_img all --edit_algo diffeditor --use_prompt --steps 8
# Evaluation SelfGuidance on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo sg --use_prompt --steps 8
python run_move_static.py --which_dataset ReS --which_img all --edit_algo sg --use_prompt --steps 8
# Evaluation Ours on Evaluation Datasets
python run_move_static.py --which_dataset COCOEE --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 1 --steps 8
python run_move_static.py --which_dataset ReS --which_img all --edit_algo ours --use_gsn 1 --inversion_free 1 --sa_masking_ipt 1 --use_copy_paste 1 --steps 8
