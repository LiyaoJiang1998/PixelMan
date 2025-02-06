# Ablations
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/null/sd1p5_16/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/null/sd1p5_16/ours_gsn-0_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-0_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/null/sd1p5_16/ours_gsn-1_invfree-0_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-0_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/null/sd1p5_16/ours_gsn-1_invfree-1_sam-0_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-0_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/null/sd1p5_16/ours_gsn-1_invfree-1_sam-1_cp-0/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-0_4-6-0.2-0.8

# Comparisons
# @16 steps
# COCOEE dataset
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/null/sd1p5_16/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/prompt/sd1p5_16/sg/4-6-0.2-0.8 --method_name sg
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/prompt/sd1p5_16/dragon/4-6-0.2-0.8 --method_name dragon
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 16 --input_path outputs/COCOEE/prompt/sd1p5_16/diffeditor/4-6-0.2-0.8 --method_name diffeditor
# ReS dataset
python scripts/evaluate_metrics.py --which_dataset ReS --steps 16 --input_path outputs/ReS/null/sd1p5_16/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset ReS --steps 16 --input_path outputs/ReS/prompt/sd1p5_16/sg/4-6-0.2-0.8 --method_name sg
python scripts/evaluate_metrics.py --which_dataset ReS --steps 16 --input_path outputs/ReS/prompt/sd1p5_16/dragon/4-6-0.2-0.8 --method_name dragon
python scripts/evaluate_metrics.py --which_dataset ReS --steps 16 --input_path outputs/ReS/prompt/sd1p5_16/diffeditor/4-6-0.2-0.8 --method_name diffeditor

# @50 steps
# COCOEE dataset
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 50 --input_path outputs/COCOEE/null/sd1p5_50/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 50 --input_path outputs/COCOEE/prompt/sd1p5_50/sg/4-6-0.2-0.8 --method_name sg
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 50 --input_path outputs/COCOEE/prompt/sd1p5_50/dragon/4-6-0.2-0.8 --method_name dragon
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 50 --input_path outputs/COCOEE/prompt/sd1p5_50/diffeditor/4-6-0.2-0.8 --method_name diffeditor
# ReS dataset
python scripts/evaluate_metrics.py --which_dataset ReS --steps 50 --input_path outputs/ReS/null/sd1p5_50/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset ReS --steps 50 --input_path outputs/ReS/prompt/sd1p5_50/sg/4-6-0.2-0.8 --method_name sg
python scripts/evaluate_metrics.py --which_dataset ReS --steps 50 --input_path outputs/ReS/prompt/sd1p5_50/dragon/4-6-0.2-0.8 --method_name dragon
python scripts/evaluate_metrics.py --which_dataset ReS --steps 50 --input_path outputs/ReS/prompt/sd1p5_50/diffeditor/4-6-0.2-0.8 --method_name diffeditor

# @8 steps
# COCOEE dataset
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 8 --input_path outputs/COCOEE/null/sd1p5_8/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 8 --input_path outputs/COCOEE/prompt/sd1p5_8/sg/4-6-0.2-0.8 --method_name sg
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 8 --input_path outputs/COCOEE/prompt/sd1p5_8/dragon/4-6-0.2-0.8 --method_name dragon
python scripts/evaluate_metrics.py --which_dataset COCOEE --steps 8 --input_path outputs/COCOEE/prompt/sd1p5_8/diffeditor/4-6-0.2-0.8 --method_name diffeditor
# ReS dataset
python scripts/evaluate_metrics.py --which_dataset ReS --steps 8 --input_path outputs/ReS/null/sd1p5_8/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8 --method_name ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8
python scripts/evaluate_metrics.py --which_dataset ReS --steps 8 --input_path outputs/ReS/prompt/sd1p5_8/sg/4-6-0.2-0.8 --method_name sg
python scripts/evaluate_metrics.py --which_dataset ReS --steps 8 --input_path outputs/ReS/prompt/sd1p5_8/dragon/4-6-0.2-0.8 --method_name dragon
python scripts/evaluate_metrics.py --which_dataset ReS --steps 8 --input_path outputs/ReS/prompt/sd1p5_8/diffeditor/4-6-0.2-0.8 --method_name diffeditor
