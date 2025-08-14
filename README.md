# [MathBook: Multi-Dimensional Rational Exploration for Incentivizing Visual Mathematical Reasoning](https://arxiv.org/pdf/2407.01284)

<div align="center"> 


![Dataset](https://img.shields.io/badge/Dataset-MathBook-green)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/papers/)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
![Mathematical Reasoning](https://img.shields.io/badge/Task-Mathematical_Reasoning-red) 
</div>

<p align="center">
    <img src="./assets/logo-another.png" width="35%"> <br>
</p>

<p align="center">
🤗 <a href="https://huggingface.co" target="_blank">MathBook-Standard</a> ｜
🤗 <a href="https://huggingface.co" target="_blank">MathBook-Pro</a> ｜
🤗 <a href="https://huggingface.co" target="_blank">MathBook-RL-7B</a>
</p>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 News
  **[2023.7.2]** Our paper is now accessible at https://arxiv.org/abs/2407.01284.
  
  **[2024.7.2]** Our dataset is now accessible at [Huggingface Datasets](https://huggingface.co/datasets/We-Math/We-Math/).
  
  **[2024.7.2]** Our project homepage can be accessed at https://we-math.github.io/.


## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Cold-Start SFT Stage](#cold-start-sft-stage)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Fine-Tuning](#2-fine-tuning)
  - [Progressive Alignment RL](#progressive-alignment-rl)
    - [1. Environment Setup](#1-environment-setup-1)
    - [2. RL Training](#2-rl-training)
      - [2.1 Pre-aligned RL (MathBook-Standard)](#21-pre-aligned-rl-mathbook-standard)
      - [2.2 Dynamic Scheduling RL (MathBook-Pro)](#22-dynamic-scheduling-rl-mathbook-pro)


## 💡 Overview
Inspired by human-like mathematical reasoning, we introduce We-Math, the first benchmark specifically designed to <b>explore the problem-solving principles beyond the end-to-end performance.</b> We meticulously collect and categorize 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and 5 layers of knowledge granularity.

<p align="center">
    <img src="assets/fig_lun.png" alt="Overview diagram and the statistics of WE-MATH" style="width: 85%;" /> <br>
    Overview diagram and the statistics of <b>We-Math</b>.
</p>

We firstly <b>decompose composite problems into sub-problems</b> according to the required knowledge concepts and introduce a novel four-dimensional metric, namely <b>Insufficient Knowledge (IK)</b>, <b>Inadequate Generalization (IG)</b>, <b>Complete Mastery (CM)</b>, and <b>Rote Memorization (RM)</b> to hierarchically assess inherent issues in LMMs’ reasoning process.

<p align="center">
    <img src="assets/3-example.png" alt="The pipeline of knowledge-based data decomposition (an example of a three-step problem in We-Math)." style="width: 65%;" /> <br>
    The pipeline of knowledge-based data decomposition (an example of a three-step problem in We-Math).
</p>

<p align="center">
    <img src="assets/metric_2.png" alt="The pipeline of knowledge-based data decomposition (left) and an example of the four-dimensional metrics for evaluating a two-step problem (right), using both loose and strict settings." style="width: 95%;" /> <br>
    An example of the four-dimensional metrics for evaluating a two-step problem, using both loose and strict settings.
</p>


With We-Math, we conduct a thorough evaluation of existing LMMs in visual mathematical reasoning and reveal a negative correlation between solving step and problem-specific performance. We confirm the IK issue of LMMs can be effectively improved via knowledge augmentation strategy. More notably, <b>the primary challenge of GPT-4o has significantly transitioned from IK to IG, establishing it as the first LMM advancing towards the knowledge generalization stage.</b> In contrast, other LMMs exhibit a marked inclination towards Rote Memorization they correctly solve composite problems involving multiple knowledge concepts, yet fail in answering sub-problems. We anticipate that We-Math will open new pathways for advancements in visual mathematical reasoning for LMMs.

<p align="center">
    <img src="assets/fig1_result.png" alt="pipeline of decomposition" style="width: 95%;" /> <br>
    Overview of LMMs' performances on We-Math. Figures from left to right illustrates the (1) accuracy of different LMMs on various problem-solving steps, (2) the performance in different visual mathematics categories and (3) the result in knowledge based reasoning evaluation.
</p>

## 🏃 Quick Start

### ❄️ Cold-Start SFT Stage

#### 1. Environment Setup

In this step, we will describe how to perform a cold start for the SFT stage using the [ms-swift](https://github.com/modelscope/ms-swift) repository. Please first set up the environment for ms-swift.

```bash
pip install ms-swift -U
```
#### 2. Fine-Tuning

Our SFT dataset consists of two parts: 200 pure text samples and 800 samples with associated images. Download the SFT dataset from 🤗[MathBook-SFT](https://huggingface.co/datasets/) and refer to the script below for fine-tuning.

```bash
nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
MAX_PIXELS=4194304 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model_type qwen2_5_vl \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --num_train_epochs 1 \
    --train_type full \
    --deepspeed zero2 \
    --tuner_backend peft \
    --torch_dtype bfloat16 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --eval_steps 1000 \
    --attn_impl flash_attn \
    --output_dir checkpoint \
    --dataset mathbook_sft.jsonl \
    --per_device_train_batch_size 1
```


### 🔥 Progressive Alignment RL

#### 1. Environment Setup
you can install our additional environment as follow:

```bash
pip install -r requirements.txt
```

#### 2. RL Training

Both RL stages are developed based on the [EasyR1](https://github.com/hiyouga/EasyR1) codebase to fit our workflow.

For data preparation, you can directly download Parquet-format datasets from [🤗MathBook-Standard](https://huggingface.co) for training.

##### 2.1 Pre-aligned RL (MathBook-Standard)

```bash
cd pre_align

python3 -m verl.trainer.main \
    config=pre_align_r1v.yaml \
```

##### 2.2 Dynamic Scheduling RL (MathBook-Pro)

```bash
cd dynamic_scheduling

python3 -m verl.trainer.main \
    config=dynamic_scheduling_r1v.yaml \
```
#### Merge Checkpoint in Hugging Face Format
```bash
python scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## 📝 Evaluation Piplines on MathBook-Eval

### Response Generation 
The models generate responses based on the given questions and images. Examples for generating responses from some LMMs are provided in the [evaluation](./evaluation). Our prompt specifies the format of answer generation to facilitate subsequent extraction of the answer using string matching. Please refer to the following template to prepare your result JSON files for subsequent evaluation.
```json
{
    "ID": "3steps_165",
    "split": "testmini",
    "knowledge concept": "Area of Circles",
    "question": "As shown in the figure, there is a circular flower bed. Mary walked from the northernmost point of the flower bed along the edge to the easternmost point, taking a total of 80 steps. It is known that Mary's average step length is 0.628 cm, what is the area of the flower bed (  ) m²?(π = 3.14)",
    "option": "A. 200.96;B. 3215.36;C. 6280;D. 32; E. No correct answer",
    "answer": "B",
    "image_path": "3steps/image/165-3.png",
    "key": "3steps_3",
    "question number": 1575,
    "knowledge concept description": "Area of ...",
    "response": "<Thought process>: ... <Answer>: ..."
}
```
### Score Calculation



## 📜 License

Our dataset are distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.


## 📄 Cite

If you find **MathBook** useful for your your research and applications, please kindly cite using this BibTeX:

```bibtex

```
## 🤝 Acknowledge

## 📞 Contact

For any questions or feedback, please reach out to us at [qrq@bupt.edu.cn](qrq@bupt.edu.cn) or [qiunatan@bupt.edu.cn](qiunatan@bupt.edu.cn).
