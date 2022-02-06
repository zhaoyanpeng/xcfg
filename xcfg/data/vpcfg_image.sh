#!/usr/bin/bash
# please read the comments and set variables as you need

if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH=$MYHOME
fi
root=$OUTPUT_PATH
echo $root $OUTPUT_PATH $1 

args=$1

export CUDA_VISIBLE_DEVICES="3"

# set dependencies

this_path=$(pwd)
git clone --branch beta https://github.com/zhaoyanpeng/vipant.git
export PYTHONPATH=$this_path/vipant:$PYTHONPATH
echo $PYTHONPATH 

# download and `tar -zxf' flickr30k-images.tar.gz from 
#   https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl
# and place it in `flickr30k', so we have `flickr30k/flickr30k-images'
flickr_root=CHANGE_ME_TO_FLICKR_PATH/flickr30k

# download and unzip 
#   http://images.cocodataset.org/zips/train2014.zip
#   http://images.cocodataset.org/zips/val2014.zip
#   http://images.cocodataset.org/zips/test2014.zip
# and place them in `mscoco', so we have `mscoco/{train2014,test2014,val2014}'
mscoco_root=CHANGE_ME_TO_MSCOCO_PATH/mscoco

# this is needed only when you want to use CLIP image encoder
clip_root=CHANGE_ME_TO_CLIP_MODEL_PATH 
clip_name=ViT-B32 #ViT-B16 #RN50x16 #

# running settings
batch_size=512 #512
peep_rate=15 #50 #
num_proc=8

create_npy="O"
if [ "$create_npy" ]; then
    ###############################
    # BEGIN create .npy from .npz files
    ###############################
    echo "create .npy files..."

    # ideally, the .npy (image vectors) should be in 
    # the same directory as the correspondiing parsing data
    flickr_out_root=CHANGE_ME_TO_FLICKR_PARSE_PATH/flickr_emnlp
    mscoco_out_root=CHANGE_ME_TO_MSCOCO_PARSE_PATH/mscoco_emnlp
    npz_token=resn-152 #clip-b32

    args="--npz_token $npz_token
    --flickr_root $flickr_root --mscoco_root $mscoco_root
    --flickr_out_root $flickr_out_root --mscoco_out_root $mscoco_out_root
    "

    echo python encode_image.py $args 
    #nohup 
    python encode_image.py $args 
    #>> ./encode_image.flickr 2>&1 &

    exit 0
fi

###############################
# encode images into .npz files
###############################
echo "create .npz files..."

args="--num_proc $num_proc --batch_size $batch_size
--flickr_root $flickr_root --mscoco_root $mscoco_root
--clip_model_root $clip_root --clip_model_name $clip_name --peep_rate $peep_rate
"

echo python encode_image.py $args 
nohup python encode_image.py $args >> ./encode_image152.flickr 2>&1 &
