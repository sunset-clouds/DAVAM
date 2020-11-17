sh seven.sh \
  -n=1 \
  --seed 783435 \
  --batch_size 64 \
  --train_tf_ratio 1.0 \
  --eval_tf_ratio 1.0 \
  --lr_start 0.1 \
  --vq_weight_max 2.0 \
  --train_data /opt/ml/disk/datasets/yahoo_data/yahoo.train.txt \
  --val_data /opt/ml/disk/datasets/yahoo_data/yahoo.valid.txt \
  --test_data /opt/ml/disk/datasets/yahoo_data/yahoo.test.txt 

#   --dataset ptb \
#   --train_data /opt/ml/disk/datasets/ptb/train.txt \
#   --val_data /opt/ml/disk/datasets/ptb/valid.txt \
#   --test_data /opt/ml/disk/datasets/ptb/test.txt 
  
#  --train_data /opt/ml/disk/datasets/yelp/train.txt \
#  --val_data /opt/ml/disk/datasets/yelp/valid.txt \
#  --test_data /opt/ml/disk/datasets/yelp/test.txt 
  
