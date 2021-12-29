LANGS=("ENGLISH") 

SPLITS=("train_caps.parsed:train.txt" "val_caps.parsed:val.txt" "test_caps.parsed:test.txt")

# MSCOCO
iroot=REPLACE_ME # root of parsed mscoco captions
oroot=./mscoco

# Flickr
iroot=REPLACE_ME # root of parsed flickr captions
oroot=./flickr

for lang in ${LANGS[@]}; do
    lang_lower=${lang,,}

    for iname_oname in ${SPLITS[@]}; do
        iname="${iname_oname%%:*}"
        oname="${iname_oname#*:}"
        
        echo $iroot/$iname
        echo $oroot/$oname

        python vpcfg.py \
            --ifile $iroot/$iname \
            --ofile $oroot/$oname 
    done
done

