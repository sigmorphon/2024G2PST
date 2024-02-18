EXPERIMENT="NAME YOUR EXPERIMENTS"
TRAIN="NAME YOUR TRAINING FILE"
DEV="NAME YOUR EVALUATION FILE"

yoyodyne-train \
	--model_dir="./" \
	--experiment "${EXPERIMENT}" \
    --train "${TRAIN}" \
    --val "${DEV}" \
    --arch "transducer" \
	--source_encoder_arch "lstm" \
    --features_encoder_arch "linear" \
    --oracle_em_epochs 10 \
    --embedding_size 256 \
    --hidden_size 1024 \
    --encoder_layers 4 \
    --decoder_layers 1 \
    --batch_size 32 \
    --dropout .1 \
    --check_val_every_n_epoch 1 \
    --log_every_n_step 2 \
    --gradient_clip_val 3 \
    --label_smoothing .1 \
    --optimizer adam \
    --learning_rate .0005 \
	--max_steps 1000 \
    --seed 49 \
    --accelerator gpu \
    --precision 16 \
	--features_col 3 \
    --check_val_every_n_epoch 1