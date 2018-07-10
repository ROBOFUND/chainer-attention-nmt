#!/bin/sh
#

SOURCE=wakati/preprocessed-summarize-source.txt
TARGET=wakati/preprocessed-summarize-target.txt
VOCAB_SOURCE=wakati/summarize-source-vocab.txt
VOCAB_TARGET=wakati/summarize-target-vocab.txt

mkdir -p result-gpu
python -u train.py -g 0 -o result-gpu \
       --max-source-sentence 200 --epoch 100 \
       $SOURCE $TARGET $VOCAB_SOURCE $VOCAB_TARGET \
       2>&1 | tee log-gpu-e100.txt
