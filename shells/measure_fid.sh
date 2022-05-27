nohup python -u ../metrics/FID/fid_score.py \
  --model_path '../data/Models/pretrained_cnn/inception_v3_google-1a9a5a14.pth' \
  --batch_size 64 \
	--gpu 0 \
  --image_path \
	--output_dir 1_21/attr_coco > 1_21/attr_coco &
