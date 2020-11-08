#!/bin/bash
LANGS=("ENGLISH" "CHINESE" "BASQUE" "GERMAN" "FRENCH" "HEBREW" "HUNGARIAN" "KOREAN" "POLISH" "SWEDISH") 

root=$1
for lang in ${LANGS[@]}; do
    lang_lower=${lang,,}
    prefix=$root$lang_lower
    echo "processing $prefix"

    train=$prefix"-train.txt"
    valid=$prefix"-valid.txt"
    test=$prefix"-test.txt"

    python batchify.py \
        --valfile $valid \
        --testfile $test \
        --trainfile $train \
        --outputfile $prefix \
        --vocabsize 10000 --lowercase 1 --replace_num 1
done

