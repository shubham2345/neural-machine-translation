#!/bin/bash

# Define paths
DATASET_DIR="dataset"
CLEANED_DIR="$DATASET_DIR/cleaned_data"
TOKENIZED_DIR="$DATASET_DIR/tokenized_data"
BPE_DIR="$DATASET_DIR/bpe_data"
BINARIZED_DIR="$DATASET_DIR/binarized_data"
MOSES_DIR="mosesdecoder"
SUBWORD_NMT_DIR="subword-nmt"
BPE_OPS=32000  # Number of BPE merges

# Create output directories
mkdir -p $TOKENIZED_DIR $BPE_DIR $BINARIZED_DIR

echo "🚀 Step 1: Tokenization using Moses tokenizer..."
for lang in fr en; do
    for split in train valid test; do
        cat $CLEANED_DIR/${split}.clean.${lang} | \
        $MOSES_DIR/scripts/tokenizer/tokenizer.perl -l ${lang} > $TOKENIZED_DIR/${split}.tok.${lang}
    done
done

echo "✅ Tokenization completed. Files saved in $TOKENIZED_DIR"

echo "🚀 Step 2: Learning BPE..."
$SUBWORD_NMT_DIR/subword_nmt/learn_bpe.py -s $BPE_OPS < $TOKENIZED_DIR/train.tok.fr > $BPE_DIR/bpe_code.fr
$SUBWORD_NMT_DIR/subword_nmt/learn_bpe.py -s $BPE_OPS < $TOKENIZED_DIR/train.tok.en > $BPE_DIR/bpe_code.en

echo "✅ BPE learning completed."

echo "🚀 Step 3: Applying BPE..."
for lang in fr en; do
    for split in train valid test; do
        $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py -c $BPE_DIR/bpe_code.${lang} < $TOKENIZED_DIR/${split}.tok.${lang} > $BPE_DIR/${split}.bpe.${lang}
    done
done

echo "✅ BPE application completed. Files saved in $BPE_DIR"

echo "🚀 Step 4: Binarization using Fairseq..."
fairseq-preprocess \
    --source-lang fr --target-lang en \
    --trainpref $BPE_DIR/train.bpe --validpref $BPE_DIR/valid.bpe --testpref $BPE_DIR/test.bpe \
    --destdir $BINARIZED_DIR \
    --workers 4 \
    --joined-dictionary

echo "✅ Binarization completed. Binarized data saved in $BINARIZED_DIR"

