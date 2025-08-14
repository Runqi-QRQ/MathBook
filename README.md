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


## Outlines
- [📣 News](https://github.com/We-Math/We-Math/blob/main/README.md#-news-)
- [💡 Overview](https://github.com/We-Math/We-Math/blob/main/README.md#-about-we-math)
- [🏆 Leaderboard on We-Math 🏆](https://github.com/We-Math/We-Math/blob/main/README.md#-leaderboard-on-we-math-)
- [📝 Evaluation Piplines on We-Math](https://github.com/We-Math/We-Math/blob/main/README.md#-evaluation-piplines-on-we-math)
- [📊 We-Math Dataset](https://github.com/We-Math/We-Math/blob/main/README.md#-we-math-dataset)
- [📜 License](https://github.com/We-Math/We-Math/blob/main/README.md#-license)
- [🤝 Contributors](https://github.com/We-Math/We-Math/blob/main/README.md#-contributors)


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

In this step, we will describe how to perform a cold start for the SFT stage using the Llama Factory repository. Please first set up the environment for Llama Factory.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```
#### 2. Fine-Tuning

Our SFT dataset consists of two parts: 200 pure text samples and 800 samples with associated images. Download the SFT dataset from 🤗[MathBook-SFT](https://huggingface.co/datasets/) and place it in `LLaMA-Factory-main/data/sft_data.json`. Define the dataset in your dataset_info.json file as shown below:

```bash
{
  "mathbook_sft_img": {
    "file_name": "data/sft_img.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "image"
    }
  },
  "mathbook_sft_text": {
    "file_name": "data/sft_text.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

Complete the path information in LLaMA-Factory-main/examples/train_full/qwen_sft_tool_star.yaml. The file content should be as follows:

```bash
example
```

After completing the information, you can fine-tune the model using the following command:
```bash
cd LLaMA-Factory-main
bash ./examples/train_full/train_sft.sh
```

### 🔥 Progressive RL Stage

#### 1. Environment Setup
you can install our additional environment as follow:

```bash
pip install -r requirements.txt
```

#### 2. RL Training

##### 2.1 Pre-aligned RL (MathBook-Standard)

```bash
pip install -r requirements.txt
```

##### 2.2 Dynamic Scheduling RL (MathBook-Pro)

```bash
pip install -r requirements.txt
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
Due to the multiple-choice question format of our dataset and the specific answer generation prompt, we use string matching to directly extract answers, which eliminates the high cost of using additional models for further answer extraction.  The extracted answer is normalized to an option letter and calculate scores on our proposed four-dimensional metrics in [four_dimensional_metrics.py](https://github.com/We-Math/We-Math/blob/main/evaluation/four_dimensional_metrics.py).
```sh
cd evaluation

python four_dimensional_metrics_refine.py \
--model_name GPT-4o \
--output_json ../output/GPT-4o.json  \
--main_results_csv_path ../result/four_dimensional_metrics.csv
```

Performences on One-Step / Two-Step / Three-Step problems and different problem domains are obtained from [accuracy.py](https://github.com/We-Math/We-Math/blob/main/evaluation/accuracy.py).

```sh
cd evaluation

python accuracy.py \
--model_name GPT-4o \
--output_json ../output/GPT-4o.json  \
--knowledge_structure_nodes_path /data/knowledge_structure_nodes.json \
```


## 📊 We-Math Dataset

### Metric for Reasoning Evaluation
Based on the decomposed multi-step problems, we further reveal the inherent issues of LMMs in problem-solving process. We feed both the M one-step sub-problems and the original problem into LMMs, and classifying the responses into four categories
1. Insufficient Knowledge (IK): Part of one-step problems contain errors, and the multi-step problem is wrong. It is reasonable because model's insufficient grasp of single knowledge concept may lead to errors in multi-step problem.
2. Inadequate Generalization (IG): One-Step problems are all correct, but the multi-step problem is incorrect. This is also considered reasonable. While LMMs are capable of understanding individual knowledge concepts, they may struggle to generalize that knowledge to solve composite problems.
3. Complete Mastery (CM): One-Step problems are all correct, and multi-step problem is also answered correctly. This result demonstrates that the model's results are both reliable and accurate.
4. Rote Memorization (RM): One-Step problems contain errors, but the multi-step problem is answered correctly, which contradicts human logical thinking. If a model can solve composite multi-step problems but fails to answer the one-step problems needed in the process, it raises doubts about the model's reliability.

### Exmaples
<details>
<summary>🔍Examples of samples.</summary>
<p align="center">
    <img src="assets/example-2.png" width="90%"> <br>
</p>
</details>

### Knowledge cards
<details>
<summary>🔍Examples of Knowledge cards.</summary>
<p align="center">
    <img src="assets/example-card.png" width="90%"> <br>
</p>
</details>


## 📜 License

Our dataset are distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.


## 📄 Cite

If you find **MathBook** useful for your your research and applications, please kindly cite using this BibTeX:

```bibtex

```
## 🤝 Acknowledge

## 📞 Contact

For any questions or feedback, please reach out to us at [qrq@bupt.edu.cn](qrq@bupt.edu.cn) or [qiunatan@bupt.edu.cn](qiunatan@bupt.edu.cn).
