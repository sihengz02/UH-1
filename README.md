# Universal Humanoid Robot Pose Learning from Internet Human Videos
International Conference on Humanoid Robots 2025, **Oral Presentation**
<!-- <p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/sihengz02/UH-1?color=green">
        <img src="https://img.shields.io/github/last-commit/sihengz02/UH-1?color=green">
    </a>
    <br/>
</p> -->
<div align="center">
<img src="./assets/teaser.png" ></img> 
</div>
<h5 align="center">
    <a href="https://usc-gvl.github.io/UH-1/">🌐 Homepage</a> | <a href="https://huggingface.co/datasets/USC-GVL/Humanoid-X">⛁ Dataset</a> | <a href="https://huggingface.co/USC-GVL/UH-1">🤗 Models</a> | <a href="https://arxiv.org/abs/2412.14172">📑 Paper</a> | <a href="https://github.com/sihengz02/UH-1">💻 Code</a>
</h5>

Code for paper [Universal Humanoid Robot Pose Learning from Internet Human Videos](https://arxiv.org/abs/2412.14172). \
Please refer to our [project page](https://usc-gvl.github.io/UH-1/) for more demonstrations and up-to-date related resources. 



## UH-1 Model: Language-conditioned Humanoid Control

### Dependencies

To establish the environment, run this code in the shell:
```shell
conda create -n UH-1 python=3.8.11
conda activate UH-1
pip install git+https://github.com/openai/CLIP.git
pip install mujoco opencv-python
```

### Preparation

Download our **text-to-keypoint** model checkpoints from [here](https://huggingface.co/USC-GVL/UH-1).

```bash
git lfs install
git clone https://huggingface.co/USC-GVL/UH-1
```

### Inference

For **text-to-keypoint** generation,

- Change the `root_path` in `inference.py` to the path of the checkpoints you just downloaded.
- Change the `prompt_list` in `inference.py` to the language prompt you what the model to generate.

- Run the following commands, and the generated humanoid motion will be stored in the `output` folder.

```bash
python inference.py
```

The generated keypoint is in this shape: `[number of frames, 34-dim keypoint]`, where the `34-dim keypoint = 27-dim DoFs joint pose value + 3-dim root position + 4-dim root orientation`.

### Visualize

Visualize these keypoints by directly setting DoFs pose,

- Change the `file_list` in `visualize.py` to the generated humaoid motion file names.
- Run the following commands, and the rendered video will be stored in the `output` folder.

```bash
mjpython visualize.py
```
If you want to do close-loop control conditioned on the generated humanoid keypoints, you need to use the goal-conditioned humanoid control policy provided below.


## Goal-conditioned Humanoid Control Policy

### Dependencies

To set up the conda environment for Isaac Gym while avoiding dependency conflicts, we chose to create a new environment.

```bash
conda create -n UH-1-rl python=3.8
conda activate UH-1-rl
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install oauthlib==3.2.2 protobuf==5.28.1

# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .

# then make sure you are at the root folder of this project 
cd rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .

pip install "torch==1.13.1" "numpy==1.23.0" pydelatin==0.2.8 wandb==0.17.5 tqdm opencv-python==4.10.0.84 ipdb pyfqmr==0.2.1 flask dill==0.3.8 gdown==5.2.0 pytorch_kinematics==0.7.4 easydict==1.13
```

Here is a sample of our training data. Due to the file size limit of Github, the data file can be downloaded [here](https://drive.google.com/drive/folders/1v6G6GsZZ41hg1CsUB6meU8QIDwqsDbN6?usp=sharing). 
Please put the data file at `motion_lib/motion_pkl/motion_data_cmu_sample.pkl`

### Inference

To play the policy with the checkpoint we've provided, try

```bash
# make sure you are at the root folder of this project 
cd legged_gym/legged_gym/scripts
python play.py 000-00 --task h1_2_mimic --device cuda:0
```

### Train from scratch

To train the goal-conditioned RL policy from scratch, try

```bash
# make sure you are at the root folder of this project 
cd legged_gym/legged_gym/scripts
python train.py xxx-xx-run_name --task h1_2_mimic --device cuda:0
```



## Humanoid-X Data Collection

For the data collection pipeline, including **Video Clip Extraction**, **3D Human Pose Estimation**, **Video Captioning**, and **Motion Retargetting**, please refer to this [README](https://github.com/sihengz02/UH-1/blob/main/README-Humanoid-X.md).



## Citation

If you find our work helpful, please cite us:

```bibtex
@article{mao2024learning,
  title={Learning from Massive Human Videos for Universal Humanoid Pose Control},
  author={Mao, Jiageng and Zhao, Siheng and Song, Siqi and Shi, Tianheng and Ye, Junjie and Zhang, Mingtong and Geng, Haoran and Malik, Jitendra and Guizilini, Vitor and Wang, Yue},
  journal={arXiv preprint arXiv:2412.14172},
  year={2024}
}
```

