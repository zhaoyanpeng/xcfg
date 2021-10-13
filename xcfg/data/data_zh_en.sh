#!/usr/bin/sh

root=./

nohup python data_zh_en.py >> $root/data_zh_en.out 2>&1 &
