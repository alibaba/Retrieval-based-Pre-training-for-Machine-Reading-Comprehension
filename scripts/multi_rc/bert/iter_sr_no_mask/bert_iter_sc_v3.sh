data_dir='/home/admin/workspace/datasets/multirc-v2'
model_dir='/home/admin/workspace/bert-base-uncased'
model_name='iter_bert_sc_v3'
reader_name='multi_rc_sent'
oss_pretrain='bert_iter_sr_wo_mask_1/pytorch_model_20000.bin'

python main_multirc.py \
   --model_name $model_name --reader_name $reader_name \
   --model_name_or_path $model_dir \
   --do_train --do_eval \
   --train_file $data_dir/train.json --dev_file $data_dir/dev.json \
   --per_gpu_train_batch_size 16 \
   --per_gpu_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-5 \
   --num_train_epochs 8.0 \
   --max_seq_length 512 \
   --max_query_length 512 \
   --output_dir experiments/multi_rc_iter_bert/iter_mcrc_v3_iter_sr_wom_0_w_pt_20k \
   --save_steps -1 \
   --logging_steps 500 \
   --save_metric em0 \
   --warmup_steps 600 --evaluate_during_training --weight_decay 0.01 --max_grad_norm 1.0 \
   --query_dropout 0.1 --cls_type 1 \
   --oss_pretrain $oss_pretrain

python main_multirc.py \
   --model_name $model_name --reader_name $reader_name \
   --model_name_or_path $model_dir \
   --do_train --do_eval \
   --train_file $data_dir/train.json --dev_file $data_dir/dev.json \
   --per_gpu_train_batch_size 16 \
   --per_gpu_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-5 \
   --num_train_epochs 8.0 \
   --max_seq_length 512 \
   --max_query_length 512 \
   --output_dir experiments/multi_rc_iter_bert/iter_mcrc_v3_iter_sr_wom_0_w_pt_20k_s33 \
   --save_steps -1 \
   --logging_steps 500 \
   --save_metric em0 \
   --warmup_steps 600 --evaluate_during_training --weight_decay 0.01 --max_grad_norm 1.0 \
   --query_dropout 0.1 --cls_type 1 \
   --oss_pretrain $oss_pretrain --seed 33

python main_multirc.py \
   --model_name $model_name --reader_name $reader_name \
   --model_name_or_path $model_dir \
   --do_train --do_eval \
   --train_file $data_dir/train.json --dev_file $data_dir/dev.json \
   --per_gpu_train_batch_size 16 \
   --per_gpu_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-5 \
   --num_train_epochs 8.0 \
   --max_seq_length 512 \
   --max_query_length 512 \
   --output_dir experiments/multi_rc_iter_bert/iter_mcrc_v3_iter_sr_wom_0_w_pt_20k_s57 \
   --save_steps -1 \
   --logging_steps 500 \
   --save_metric em0 \
   --warmup_steps 600 --evaluate_during_training --weight_decay 0.01 --max_grad_norm 1.0 \
   --query_dropout 0.1 --cls_type 1 \
   --oss_pretrain $oss_pretrain --seed 57

python main_multirc.py \
   --model_name $model_name --reader_name $reader_name \
   --model_name_or_path $model_dir \
   --do_train --do_eval \
   --train_file $data_dir/train.json --dev_file $data_dir/dev.json \
   --per_gpu_train_batch_size 16 \
   --per_gpu_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-5 \
   --num_train_epochs 8.0 \
   --max_seq_length 512 \
   --max_query_length 512 \
   --output_dir experiments/multi_rc_iter_bert/iter_mcrc_v3_iter_sr_wom_0_w_pt_20k_s67 \
   --save_steps -1 \
   --logging_steps 500 \
   --save_metric em0 \
   --warmup_steps 600 --evaluate_during_training --weight_decay 0.01 --max_grad_norm 1.0 \
   --query_dropout 0.1 --cls_type 1 \
   --oss_pretrain $oss_pretrain --seed 67

