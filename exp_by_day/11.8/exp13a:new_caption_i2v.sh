bash scripts/train.sh      \
        -y i2v \
       -S multicam \
       -s /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train \
       -m metadata/output_recam.csv  -w exp13a -b 8 -c "2,3,4,5,6,7"  -g 92 -v 12 -i 200 -t 0.5 