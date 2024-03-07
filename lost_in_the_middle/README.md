The folder `data_position_info` indicates the relative position ratios of the answers of each sample in the context for four datasets.

Run `eval_pos.sh` to obtain the scores of the model at different positions. The results will be stored in `eval_position_results`. Before running this script, make sure you have obtained the model's inference results using `inference.sh` in the previous level.

Then, run `plot.ipynb` to obtain the results for Figure 2 of the paper.