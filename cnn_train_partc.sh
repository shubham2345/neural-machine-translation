# #!/bin/bash

# # Ensure the script stops on errors
# set -e

# # Define directories (Separate from CNN & Part A)
# DATA_BIN="data-bin/fr-en-bpe-tuned"
# SAVE_DIR="checkpoints/transformer_model_partb"

# # Ensure directories exist
# mkdir -p $SAVE_DIR
# mkdir -p prep/temp

# echo "===== Step 1: Preprocessing Data with BPE ====="

# # Learn BPE on concatenated text
# BPE_OPS=40000  # Tune this value
# python subword-nmt/learn_bpe.py -s $BPE_OPS < prep/temp/train.fr-en > prep/code_$BPE_OPS

# # Apply BPE
# for lang in fr en; do
#   for split in train dev tst; do
#     echo "Applying BPE to $split.$lang with $BPE_OPS merges..."
#     python subword-nmt/apply_bpe.py -c prep/code_$BPE_OPS < prep/temp/$split.$lang > prep/temp/bpe.$split.$lang
#   done
# done

# # Preprocess with updated BPE and a joined dictionary
# fairseq-preprocess --source-lang fr --target-lang en \
#     --trainpref prep/temp/bpe.train --validpref prep/temp/bpe.dev --testpref prep/temp/bpe.tst \
#     --destdir $DATA_BIN \
#     --workers 20 \
#     --joined-dictionary

# echo "===== Preprocessing Completed Successfully ====="

# echo "===== Step 2: Training Transformer Model ====="

# # Train Transformer Model
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train \
#     $DATA_BIN \
#     --source-lang fr --target-lang en \
#     --arch transformer --share-all-embeddings \
#     --encoder-layers 10 --decoder-layers 10 \
#     --encoder-embed-dim 512 --decoder-embed-dim 512 \
#     --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.5 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
#     --dropout 0.5 --attention-dropout 0.3 \
#     --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 12, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --max-epoch 40 \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --save-dir $SAVE_DIR



# DATA_BIN="data-bin/fr-en-bpe-exp"
# SAVE_DIR="checkpoints/cnn_model_partc"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train \
#     $DATA_BIN \
#     --source-lang fr --target-lang en \
#     --arch fconv \
#     --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --max-epoch 1 --batch-size 5000 \
#     --encoder-embed-dim 1024 \
#     --decoder-embed-dim 1024 \
#     --decoder-layers 4 \
#     --encoder-layers 4 \
#     --activation-dropout 0.1 \
#     --attention-dropout 0.1 \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --save-dir $SAVE_DIR

# # Generate translations on the test set
# fairseq-generate $DATA_BIN \
#     --path $SAVE_DIR/checkpoint_best.pt \
#     --batch-size 128 --beam 5 --remove-bpe \
#     --scoring sacrebleu \
#     > outputs/cnn_output_partc.out

# echo "Training and evaluation for Part 3 completed. Check $SAVE_DIR for model checkpoints."

#!/bin/bash

# Ensure the script stops on errors
set -e

# Define paths
DATA_BIN="data-bin/fr-en-bpe-exp"
SAVE_DIR="checkpoints/cnn_model_partc"

# Ensure directories exist
mkdir -p $SAVE_DIR

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train \
#     $DATA_BIN \
#     --source-lang fr --target-lang en \
#     --arch fconv \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
#     --lr 5e-4 --lr-scheduler reduce_lr_on_plateau \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --max-epoch 30 --batch-size 5000 \
#     --encoder-embed-dim 1024 \
#     --decoder-embed-dim 1024 \
#     --decoder-layers 6 \
#     --encoder-layers 6 \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --save-dir $SAVE_DIR

DATA_BIN="data-bin/fr-en-bpe-exp"
SAVE_DIR="checkpoints/cnn_model_partc"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train \
#     $DATA_BIN \
#     --source-lang fr --target-lang en \
#     --arch fconv \
#     --encoder-embed-dim 768 --decoder-embed-dim 768 \
#     --encoder-layers "[(768, 5)] * 24" \
#     --decoder-layers "[(768, 5)] * 24" \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 \
#     --lr-scheduler reduce_lr_on_plateau \
#     --clip-norm 0.1 \
#     --max-tokens 4096 \
#     --batch-size 2048 \
#     --dropout 0.4 \
#     --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
#     --fp16 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 10, "max_len_a": 1.3, "max_len_b": 15}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --max-epoch 35 \
#     --ddp-backend=legacy_ddp \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --save-dir $SAVE_DIR
CUDA_VISIBLE_DEVICES=2 fairseq-train \
$DATA_BIN \
--arch fconv_iwslt_de_en --dropout 0.4 \
--encoder-embed-dim 512 --decoder-embed-dim 512 --encoder-layers '[(512, 3)] * 10' \
--decoder-layers '[(512, 3)] * 10' \
--max-tokens 8192 --update-freq 2 \
--optimizer adam --lr 0.01 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--clip-norm 0.1 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--fp16-no-flatten-grads --max-epoch 10 \
--save-dir $SAVE_DIR


# Generate translations on the test set
fairseq-generate $DATA_BIN \
    --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \  # Keep if BPE was used
    --scoring sacrebleu \
    > outputs/cnn_output_partc.out

echo "Training and evaluation for Part 3 completed. Check $SAVE_DIR for model checkpoints."


# echo "===== Transformer Model Training Completed ====="

# echo "===== Step 3: Generating Translations ====="

# # Generate Translations
# fairseq-generate $DATA_BIN \
#     --path $SAVE_DIR/checkpoint_best.pt \
#     --batch-size 128 --beam 12 --remove-bpe \
#     --scoring sacrebleu \
#     > outputs/transformer_output_partb.out

# echo "===== Transformer Model Inference Completed! Check 'outputs/transformer_output_partb.out' for results ====="
