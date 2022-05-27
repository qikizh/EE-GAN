nohup python -u train_coco.py \
	--gpu 0,1,2,3 \
	--batchSize 64 \
	--output_dir 1_21/attr_coco > 1_21/attr_coco &
