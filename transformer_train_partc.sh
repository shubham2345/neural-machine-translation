#!/bin/bash

# Ensure the script stops on errors
set -e

# Define paths for the experiment
BPE_OPS=40000  # Set number of merge operations for BPE (tune this)
BPE_DIR="prep/bpe_experiments"
DATA_BIN="data-bin/fr-en-bpe-exp"
SAVE_DIR="checkpoints/transformer_model_partc"

# Ensure directories exist
mkdir -p $BPE_DIR/temp
mkdir -p $SAVE_DIR

# Learn BPE on concatenated training data (new path)
echo "Learning BPE with $BPE_OPS merges..."
cat prep/temp/train.fr prep/temp/train.en > $BPE_DIR/bpe.train.fr-en
python subword-nmt/learn_bpe.py -s $BPE_OPS < $BPE_DIR/bpe.train.fr-en > $BPE_DIR/code_$BPE_OPS

# Apply BPE to train, dev, and test sets (new path)
for lang in fr en; do
  for split in train dev tst; do
    echo "Applying BPE to $split.$lang..."
    python subword-nmt/apply_bpe.py -c $BPE_DIR/code_$BPE_OPS < prep/temp/$split.$lang > $BPE_DIR/temp/bpe.$split.$lang
  done
done

# Preprocess data with new BPE tokenized files (new path)
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $BPE_DIR/temp/bpe.train --validpref $BPE_DIR/temp/bpe.dev --testpref $BPE_DIR/temp/bpe.tst \
    --destdir $DATA_BIN \
    --workers 20 \
    --joined-dictionary 

# Train Fairseq Transformer with BPE-experimented dataset
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train \
    $DATA_BIN \
    --source-lang fr --target-lang en \
    --arch transformer --share-all-embeddings \
    --encoder-layers 8 --decoder-layers 8 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.5 \
    --lr 2e-4 --lr-scheduler inverse_sqrt --warmup-updates 8000 \
    --dropout 0.4 --attention-dropout 0.2 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.05 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 10, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --max-epoch 35 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $SAVE_DIR

# Generate translations on the test set
fairseq-generate $DATA_BIN \
    --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu \
    > outputs/transformer_output_partc.out

echo "Training and evaluation for Part 3 completed. Check $SAVE_DIR for model checkpoints."
