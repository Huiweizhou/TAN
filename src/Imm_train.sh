python3 -m train \
	--data_path ../data/Immunotherapy \
	--log_path ./logs/Imm_TAN \
	--model_save_path ./Models/Imm_TAN \
	--device cuda:0 \
	--model TAN \
	--agg_type gcn \
	--backbone Sage \
	--test_type full \
	--batch_size 2048 \
	--nhead 8 \
	--num_layers 2 \
	--input_dim 300 \
	--hidden_dim 256 \
	--output_dim 1 \
	--learning_rate 5e-4 \
	--dropout_rate 0.5 \
	--epoch 20 \
	--sample_size_1 -1 \
	--sample_size_2 -1 \
	--attn_wnd 6 \
	--year_start 1950 \
	--year_end 2010 \
	--year_interval 10

python3 -m train \
	--data_path ../data/Immunotherapy \
	--log_path ./logs/Imm_MTSa \
	--model_save_path ./Models/Imm_MTSa \
	--device cuda:0 \
	--model MTSa \
	--agg_type gcn \
	--backbone Sage \
	--test_type full \
	--batch_size 2048 \
	--nhead 8 \
	--num_layers 2 \
	--input_dim 300 \
	--hidden_dim 256 \
	--output_dim 1 \
	--learning_rate 5e-4 \
	--dropout_rate 0.5 \
	--epoch 20 \
	--sample_size_1 -1 \
	--sample_size_2 -1 \
	--attn_wnd 6 \
	--year_start 1950 \
	--year_end 2010 \
	--year_interval 10

python3 -m train \
	--data_path ../data/Immunotherapy \
	--log_path ./logs/Imm_TDa \
	--model_save_path ./Models/Imm_TDa \
	--device cuda:0 \
	--model TDa\
	--agg_type gcn \
	--backbone Sage \
	--test_type full \
	--batch_size 2048 \
	--nhead 8 \
	--num_layers 2 \
	--input_dim 300 \
	--hidden_dim 256 \
	--output_dim 1 \
	--learning_rate 5e-4 \
	--dropout_rate 0.5 \
	--epoch 20 \
	--sample_size_1 -1 \
	--sample_size_2 -1 \
	--attn_wnd 6 \
	--year_start 1950 \
	--year_end 2010 \
	--year_interval 10



