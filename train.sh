fairseq-train dataset/binarized_data \
    --arch fconv \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-layers "[(512, 3)] * 20" --decoder-layers "[(512, 3)] * 20" \
    --optimizer adam --lr 0.25 --lr-scheduler reduce_lr_on_plateau \
    --clip-norm 0.1 \
    --max-tokens 4096 --fp16 \
    --save-dir checkpoints/cnn_model

fairseq-train dataset/binarized_data \
    --arch transformer \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 0.0005 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --clip-norm 0.1 \
    --dropout 0.3 \
    --max-tokens 4096 --fp16 \
    --save-dir checkpoints/transformer_model

