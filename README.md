# Tacotron 2
Implementation of [Tacotron 2](https://arxiv.org/pdf/1712.05884.pdf) TTS model in PyTorch

## Usage

### Setup
To launch and inference in nvidia-docker container follow these instructions:

0. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
1. Run `./docker-build.sh`

### Training
To launch training follow these instructions:

1. Set preferred configurations in `config/config.yaml` or leave it as it is
2. In `docker-run.sh` change `memory`, `memory-swap`, `shm-size`, `cpuset-cpus`, `gpus`, and data `volume` to desired values
3. Set WANDB_API_KEY environment variable to your wandb key
4. Run `./docker-train.sh`

All outputs including models will be saved to `outputs` dir.

### Inference
To launch inference run the following command:
```
./docker-inference.sh model_path label_encoder_path waveglow_model_path device input_text
```
Where:
* `model_path` is a path to .ckpt model file
* `label_encoder_path` is a path to .pickle label encoder file. It is generated during training by `fut_label_encoder.py` script
* `waveglow_model_path` is a path to wavegkiw .pt model file. It can be downloaded [here](https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view) (Link from https://github.com/NVIDIA/waveglow)
* `device` is the device to inference on: either 'cpu', 'cuda' or cuda device number
* `input_text` is an input text for TTS

Predicted output wav and spectrogram will be saved in `inferenced` folder

## Pretrained models
All pretrained files for inference (tacotron 2 checkpoint trained on LJSpeech, label encoder and waveglow checkpoint) can be downloaded [here](https://drive.google.com/drive/folders/1f9sqm9-8zU5Z4J7wLelLMHVRJmTiemmm?usp=sharing).
