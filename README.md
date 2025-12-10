# OpenVLA: An Open-Source Vision-Language-Action Model

[**官方文档**](https://openvla.github.io/) | [**复现环境**](#复现环境) | [**LIBERO**](#LIBERO)


<hr style="border: 2px solid gray;"></hr>

## 官方文档

[![Homepage](https://img.shields.io/badge/Homepage-blue?style=for-the-badge)](https://openvla.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```

## 复现环境配置

复现结果使用以下设备及环境：

GPU: NVIDIA Tesla A100

硬件架构：| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |

[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)

```bash
conda create -n openvla python=3.10 -y
conda activate openvla

# 原论文指出该项目要满足Python 3.10.13、PyTorch 2.2.0、transformers 4.40.1 和 flash-attn 2.5.5的环境
# 原版的环境，CUDA 12.4
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  
# CUDA 12.1
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# CUDA 11.8
# conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

git clone https://github.com/ZYangChen/openvla.git
cd openvla
cd dlimp_openvla
pip install -e .
# pip uninstall sympy -y #若有环境冲突，可尝试该过程
# pip install sympy==1.13.1
cd ..
pip install -e . --no-deps
pip install -r requirements-vla.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

如果用Conda环境安装报错libtorch_cpu.so中缺少了iJIT_NotifyEvent这个符号，可考虑改用wheel安装环境。但是wheel安装环境后按如上步骤配置环境，可能出现版本冲突问题，这一问题可以参考我在[issue #1 回答中的步骤](https://github.com/ZYangChen/openvla/issues/1#issuecomment-3635885130)解决。


<hr style="border: 2px solid gray;"></hr>

## LIBERO

### 环境

在已有环境上进一步配置

```bash
pip install -r experiments/robot/libero/libero_requirements.txt #在openvla目录下
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

apt-get install libegl1 mesa-utils libgl1-mesa-glx
```

也可以考虑手动下载[LIBERO库](https://github.com/Lifelong-Robot-Learning/LIBERO)。

### 数据集

下载地址：https://libero-project.github.io/datasets

### 权重

```bash
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

# 下载fine-tuned OpenVLA via LoRA (r=32) on four LIBERO task suites independently: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
huggingface-cli download --resume-download openvla/openvla-7b-finetuned-libero-spatial --local-dir ./weight/libero/openvla-7b-finetuned-libero-spatial

```

The four checkpoints are available on Hugging Face:
* [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
* [openvla/openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)
* [openvla/openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)
* [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)

### 推理

#### e.g. LIBERO-Spatial

数据集路径可以在推理时设置，也可以直接修改config.yaml

```bash
# Launch LIBERO-Spatial evals
#CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint weight/libero/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_wandb False \
  --center_crop True

# Do you want to specify a custom path for the dataset folder? (Y/N):
Y

# Please enter the path to the dataset folder:
/remote-home/path_to_your_root/vla/openvla/datasets # 你的数据集路径 <PATH TO DATASET FOLDER>
```

或者

```bash
vim /root/.libero/config.yaml

datasets: /remote-home/path_to_your_root/vla/openvla/datasets
```

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy from scratch | 78.3 ± 1.1% | **92.5 ± 0.7%** | 68.3 ± 1.2% | 50.5 ± 1.3% | 72.4 ± 0.7% |
| Octo fine-tuned | 78.9 ± 1.0% | 85.7 ± 0.9% | **84.6 ± 0.9%** | 51.1 ± 1.3% | 75.1 ± 0.6% |
| OpenVLA fine-tuned | **84.7 ± 0.9%** | 88.4 ± 0.8% | 79.2 ± 1.0% | **53.7 ± 1.3%** | **76.5 ± 0.6%** |

Each success rate is the average over 3 random seeds x 500 rollouts each (10 tasks x 50 rollouts per task).

更新中...


