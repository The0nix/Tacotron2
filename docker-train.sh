set -e
tmp_dir=$(mktemp -d -t inference-XXXXXXXXXX)
cp $1 $tmp_dir  # Copy waveglow model

docker run \
	-it \
	--memory=8g \
	--memory-swap=2g \
	--shm-size=8g \
	--cpuset-cpus=0-11 \
	--gpus '"device=0"' \
	--volume /street/data:/home/user/data \
	--volume $(pwd)/config:/home/user/config \
	--volume $(pwd)/outputs:/home/user/outputs \
	--volume $tmp_dir:/home/user/files \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	tacotron2-tamerlan-tabolov \
	bash -c "
	  python ./src/fit_label_encoder.py && \
    python ./src/train.py \
    model.vocoder_checkpoint_path=files/$(basename $1)
	" || rm -rf $tmp_dir
