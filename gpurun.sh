#!/bin/sh
#

SOURCE=wakati/preprocessed-summarize-source.txt
TARGET=wakati/preprocessed-summarize-target.txt
VOCAB_SOURCE=wakati/summarize-source-vocab.txt
VOCAB_TARGET=wakati/summarize-target-vocab.txt

python -u train.py -g 0 -o result-cpu \
       --max-source-sentence 200 \
       $SOURCE $TARGET $VOCAB_SOURCE $VOCAB_TARGET
