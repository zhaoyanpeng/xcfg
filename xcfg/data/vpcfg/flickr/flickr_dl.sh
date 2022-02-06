#wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/test_2016_flickr.txt -O test_ids.txt
#wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/train.txt -O train_ids.txt
#wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/val.txt -O val_ids.txt

wget http://hockenmaier.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz
tar -zxvf flickr30k.tar.gz
rm flickr30k.tar.gz
