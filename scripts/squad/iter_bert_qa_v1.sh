SQUAD_DIR='../datasets/squad'

python run_squad.py \
  --model_type bert \
  --model_name_or_path experiments/squad_v1_bert_iter/iter_w_pt1_20k \
  --do_eval \
  --version_2_with_negative \
  --do_lower_case \
  --data_dir $SQUAD_DIR \
  --train_file train-v1.1.json \
  --predict_file dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir experiments/squad_v1_bert_iter/iter_w_pt1_20k