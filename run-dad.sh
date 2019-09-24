#--image_type depth/IR/D+IR
#--view top/front
python main.py --root_path ~/ \
	--video_path ~/DAD \
	--annotation_path MyRes3D_AnoDect/annotation_Dad/val12456.json \
	--result_path MyRes3D_AnoDect/results \
	--dataset dad \
	--view top \
	--image_type D+IR \
	--n_classes 2 \
	--model shufflenet \
	--groups 3 \
	--width_mult 1.0 \
	--train_crop random \
	--ft_portion complete \
	--learning_rate 0.01 \
	--sample_duration 32 \
	--downsample 1 \
	--batch_size 10 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
	--pretrain_path MyRes3D_ad/results/kinetics_shufflenet_1.0x_G3_RGB_16_best.pth \
	# --no_train \
	# --resume_path MyRes3D_AE/results/Table5/mv2_front_depth_0916/dad_mobilenetv2_1.0x_RGB_32_best.pth \
    # --resume_path MyRes3D_AE/results/0.5top_0812_IR/dad_shufflenet_0.5x_RGB_32_best.pth \
    # \
    ##
	# --resume_path MyRes3D/report/jester_squeezenet_v11_RGB_32_best_090200.pth \
	# --resume_path MyRes3D/report/jester_schrinknet_RGB_16_smallest_noShuffle_epoch37_07820.pth \
	# --resume_path MyRes3D/results/jester_schrinknet_RGB_16_checkpoint.pth \
	# --version 1.1 \
	# --width_mult 0.5 \
