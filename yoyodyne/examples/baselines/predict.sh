# Prediction data
ARCH="transducer"
TEST='../../../2024G2PST/data/tsv/latin/latin_dev_label.tsv' ##"TEST DATA HERE"

# Trained model
EXPERIMENT="NAME YOUR EXPERIMENTS"
CHECKPOINT="version_5/checkpoints/model-epoch=002-val_accuracy=0.399.ckpt"  #"PATH TO CHECKPOINT"

# Prediction args
OUTPUT="PREDICTIONS FILE HERE"

yoyodyne-predict \
    --predict "${TEST}" \
    --output "${OUTPUT}" \
    --model_dir="./" \
    --experiment="${EXPERIMENT}" \
    --checkpoint="${EXPERIMENT}/${CHECKPOINT}" \
    --arch "${ARCH}" \
    --batch_size 1 \
    --accelerator gpu \
    --precision 16 \
	--features_col 3