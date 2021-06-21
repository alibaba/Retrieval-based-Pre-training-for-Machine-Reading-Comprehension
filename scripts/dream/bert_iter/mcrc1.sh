data_dir='../datasets/dream/data/'
model_dir='pretrained_model/bert-base-uncased'
model_name='iter_bert_mcrc'
reader_name='race_sent'

# python main_multirc.py \
#   --model_name $model_name --reader_name $reader_name \
#   --model_name_or_path $model_dir \
#   --do_train \
#   --do_eval \
#   --train_file $data_dir/train_in_race_split.json --dev_file $data_dir/dev_in_race_split.json \
#   --test_file $data_dir/test_in_race_split.json \
#   --per_gpu_train_batch_size 6 \
#   --per_gpu_eval_batch_size 6 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 8.0 \
#   --max_seq_length 512 \
#   --max_query_length 512 \
#   --output_dir experiments/dream_iter_bert/iter_mcrc_1_0_wo_pt \
#   --save_steps -1 \
#   --logging_steps 200 \
#   --save_metric accuracy \
#   --warmup_steps 200 --evaluate_during_training --weight_decay 0.01 --max_grad_norm 5.0 --num_workers 8 \
#   --query_dropout 0.1 --cls_type 1


python main_multirc.py \
  --model_name $model_name --reader_name $reader_name \
  --model_name_or_path $model_dir \
  --do_train \
  --do_eval \
  --train_file $data_dir/train_in_race_split.json --dev_file $data_dir/dev_in_race_split.json \
  --test_file $data_dir/test_in_race_split.json \
  --per_gpu_train_batch_size 6 \
  --per_gpu_eval_batch_size 6 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 8.0 \
  --max_seq_length 512 \
  --max_query_length 512 \
  --output_dir experiments/dream_iter_bert/iter_mcrc_1_1_wo_pt \
  --save_steps -1 \
  --logging_steps 200 \
  --save_metric accuracy \
  --warmup_steps 200 --evaluate_during_training --weight_decay 0.01 --max_grad_norm 5.0 --num_workers 8 \
  --query_dropout 0.1 --cls_type 1
