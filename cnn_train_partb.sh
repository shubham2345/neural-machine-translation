DATA_BIN="data-bin/fr-en-no-bpe"
SAVE_DIR="checkpoints/cnn_model_partb"

CUDA_VISIBLE_DEVICES=1,5 fairseq-train \
    $DATA_BIN \
    --source-lang fr --target-lang en \
    --arch fconv  \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-layers "[(512, 3)] * 20" --decoder-layers "[(512, 3)] * 20" \
    --optimizer adam --lr 1e-4 --lr-scheduler reduce_lr_on_plateau \
    --clip-norm 0.05 \
    --max-tokens 4096 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-print-samples \
    --max-epoch 50 \
    --ddp-backend=legacy_ddp \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $SAVE_DIR

echo "CNN Model Training Completed!"

# Generate Translations
fairseq-generate $DATA_BIN \
    --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu \
    > outputs/cnn_output_partb.out

echo "CNN Model Inference Completed!"
