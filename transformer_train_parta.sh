# # fairseq-preprocess --source-lang fr --target-lang en \
# #     --trainpref prep/temp/bpe.train --validpref prep/temp/bpe.dev --testpref prep/temp/bpe.tst \
# #     --destdir data-bin/fr-en \
# #     --workers 20


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train \
#     data-bin/fr-en \
#     --source-lang fr --target-lang en \
#     --arch transformer --share-decoder-input-output-embed \
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
#     --max-epoch 30 \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --save-dir checkpoints/transformer_model_parta

# fairseq-generate data-bin/fr-en \
#     --path checkpoints/transformer_model_parta/checkpoint_best.pt \
#     --batch-size 128 --beam 5 --remove-bpe \
#     --scoring sacrebleu \
#     > outputs/transformer_output_parta.out

#!/bin/bash

# Ensure the script stops on errors
set -e

# Define the directories
DATA_BIN="data-bin/fr-en"
SAVE_DIR="checkpoints/transformer_model_parta"

# Ensure the checkpoint directory exists
mkdir -p $SAVE_DIR

fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref prep/temp/bpe.train --validpref prep/temp/bpe.dev --testpref prep/temp/bpe.tst \
    --destdir data-bin/fr-en \
    --workers 20

# Run Fairseq training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train \
    $DATA_BIN \
    --source-lang fr --target-lang en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --max-epoch 30 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $SAVE_DIR

# Generate translations on the test set
fairseq-generate $DATA_BIN \
    --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu \
    > outputs/transformer_output_parta.out

echo "Training and evaluation completed. Check $SAVE_DIR for model checkpoints."