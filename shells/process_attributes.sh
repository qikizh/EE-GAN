nohup python -u ../prepare_attributes.py \
	--gpu 0,1,2,3 \
	--save_path \
	--batchSize 64 \
	--output_dir 1_21/attr_coco > 1_21/attr_coco &
