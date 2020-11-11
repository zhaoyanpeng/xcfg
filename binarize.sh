LANGS=("ENGLISH" "CHINESE" "BASQUE" "GERMAN" "FRENCH" "HEBREW" "HUNGARIAN" "KOREAN" "POLISH" "SWEDISH") 
SPLITS=("train:train" "valid:val" "test:test")

iroot=$1
binarize=$2
if [ -z "$binarize" ]; then
    suffix=".json"
else
    suffix=".bin"
fi

for lang in ${LANGS[@]}; do
    lang_lower=${lang,,}

    #if [ "$lang" != "SWEDISH" ]; then
    #    continue
    #fi
    #echo "processing "$iroot$lang_lower

    for iname_oname in ${SPLITS[@]}; do
        iname="${iname_oname%%:*}"
        oname="${iname_oname#*:}"
        
        #echo $iroot$lang_lower"-"$iname".txt"
        #echo $iroot$lang_lower"-"$oname$suffix
        #continue

        python binarize.py \
            --ifile $iroot$lang_lower"-"$iname".txt" \
            --ofile $iroot$lang_lower"-"$oname$suffix \
            --binarize "$binarize"
    done
done
