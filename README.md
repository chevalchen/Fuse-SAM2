<div align="center">
  <img align="left" width="100" height="100" src="assets/logo.png" alt="SANSA logo">

# SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation 

<p align="center">
  <a href="https://arxiv.org/abs/2505.21795" title="Read the paper on arXiv">
    <img src="https://img.shields.io/badge/arXiv-2505.21795-b31b1b?style=flat-square&logo=arxiv&logoColor=white"
         alt="arXiv" style="vertical-align: middle;">
  </a>
  <a href="https://claudiacuttano.github.io/SANSA/" title="Open the project page">
    <img src="https://img.shields.io/badge/Project-Page-blue"
         alt="Project Page" style="vertical-align: middle;">
  </a>
  <a href="https://colab.research.google.com/github/ClaudiaCuttano/SANSA/blob/main/sansa_demo.ipynb" title="Open in Google Colab">
    <img src="https://img.shields.io/badge/Colab-Open-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"
         alt="Open in Colab" style="vertical-align: middle;">
  </a>
</p>



[Claudia Cuttano*](https://scholar.google.it/citations?user=W7lNKNsAAAAJ&hl=en) ·
[Gabriele Trivigno*](https://scholar.google.com/citations?user=JXf_iToAAAAJ&hl=en) ·
[Giuseppe Averta](https://scholar.google.it/citations?user=i4rm0tYAAAAJ&hl=en) ·
[Carlo Masone](https://scholar.google.it/citations?user=cM3Iz_4AAAAJ&hl=en)

✨ **NeurIPS 2025 Spotlight** ✨ 
</div>


SANSA unlocks the hidden semantics of **Segment Anything 2**, turning it into a **powerful few-shot segmenter** for both **objects** and **parts**.  
🚀 **No fine-tuning of SAM2 weights.**  
🧠️ **Fully promptable: points · boxes · scribbles · masks, making it ideal for real-world labeling**.  
📈 **State-of-the-art on few-shot object & part segmentation benchmarks.**  
⚡ **Lightweight: 3–5× faster, 4–5× smaller!**  


https://github.com/user-attachments/assets/b8c81a27-d8d5-496d-ae3e-eaefd5a7cf90




---

## ⚙️ Environment Setup  
To get started, create a Conda environment and install the required dependencies.
SANSA is compatible with any **PyTorch ≥ 2.0**.
The experiments in the paper were run with **PyTorch 2.7.1 (with CUDA 12.6)**, which we provide as a reference configuration. 
To set up the environment using Conda, run:  

```
conda create --name sansa python=3.10 -y
conda activate sansa
pip install -r requirements.txt
```
---

## 💡 **Getting Started**

In this repository, you will find:   
> **1. SANSA Universal Model**: a single model, fully promptable (points · boxes · scribbles · masks), for both objects & parts.  
> &nbsp;&nbsp;&nbsp; · We release this model on **TorchHub**, and include an **interactive demo** to try it on your own data.  
> &nbsp;&nbsp;&nbsp; · *Note*: this is *not* the model used for the paper benchmarks.  
> **2. Paper Results & Training**: strict few-shot and in-context benchmarks, with results and training scripts for reproducibility.

---
## 1. SANSA Universal Model 🌐
_Run on your own data (objects & parts, promptable with points · boxes · scribbles · masks)._

#### Quick Links: 📥 **[Download Weights](https://drive.google.com/file/d/1nPOdRfMfo3MQRSi1qkPEri7Gl6FCEVHe)** · 🧑‍💻 **[Interactive Notebook](https://colab.research.google.com/github/ClaudiaCuttano/SANSA/blob/main/sansa_demo.ipynb)** · 📦 [**TorchHub**](#torchhub)
  

---
### 🧑‍💻 Interactive Demo (Colab)  
Curious about SANSA? The **[Notebook](https://colab.research.google.com/github/ClaudiaCuttano/SANSA/blob/main/sansa_demo.ipynb)** lets you try it out. Mark **an object or part in one image** (point, box, scribble, or mask), and SANSA will segment the same class in the following images.   
💡 Example: draw a quick box around a person, and SANSA finds all the people in the next images.  

<p align="center">
  <img src="assets/sansa_promptable.gif" alt="Demo GIF" width="600">
</p>

----
<a id="torchhub"></a>
### 📦 TorcHub
Below is a minimal example showing how to load SANSA from TorchHub and run inference.
Use point, box, or mask prompts depending on your application.
<details>
<summary><strong>Expand for 'def format_prompt' function</strong></summary>

```python
def format_prompt(n_shots: int, prompt_input, prompt_type: str, device: torch.device = torch.device('cuda')):
    """
    Format prompt to be fed to the SANSA model. Alternatively, import as 'from util.demo_sansa import format_prompt'
    """
    assert prompt_type in ['mask', 'point', 'box']
    prompt_dict = {0: {}, 'shots': n_shots}
    prompt_d = prompt_input
    if prompt_type in ['point']:
        pts = torch.as_tensor(prompt_input, dtype=torch.float32, device=device).view(-1, 2)
        prompt_d = {'point_coords': pts.view(1, -1, 2),
                    'point_labels': torch.ones(1, pts.shape[0], dtype=torch.int32, device=device)}
    elif prompt_type == 'box':
        b = torch.as_tensor(prompt_input, dtype=torch.float32, device=device).view(-1, 4)
        x0y0 = torch.minimum(b[:, :2], b[:, 2:])
        x1y1 = torch.maximum(b[:, :2], b[:, 2:])
        point_coords = torch.stack([x0y0, x1y1], dim=1).view(1, -1, 2)
        n = point_coords.shape[1] // 2
        point_labels = torch.tensor([2, 3], dtype=torch.int32, device=device).repeat(1, n)
        prompt_d = {'point_coords': point_coords, 'point_labels': point_labels}
    prompt_dict[0][0] = {'prompt_type': prompt_type, 'prompt': prompt_d}
    return prompt_dict
```
</details> 


```python
import torch
from torchvision import transforms 
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

sup_img_path='assets/demo/images_demo/image011.jpg'
q_img_path='assets/demo/images_demo/image005.jpg'
sup_mask_path = 'assets/demo/masks_demo/image011_dog.png'

device = torch.device('cuda')
_transform = transforms.Compose([transforms.Resize(size=(640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
sup_t, q_t = _transform(Image.open(sup_img_path)), _transform(Image.open(q_img_path))
video = torch.stack([sup_t, q_t], dim=0)[None].to(device)      # [1, 2, 3, H, W]

model = torch.hub.load('ClaudiaCuttano/SANSA', 'sansa', pretrained=True, trust_repo=True, device=device)

point = np.array([[320, 330]], dtype=np.float32)  # points
#box = np.array([300, 560, 440, 320], dtype=np.float32)  # box
#mask = TF.to_tensor(Image.open(sup_mask_path).convert("L").resize((img_size, img_size), Image.NEAREST))[None].to(device) # mask

point_prompt = format_prompt(n_shots=1, prompt_input=point, prompt_type='point', device=device)
#box_prompt = format_prompt(n_shots=1, prompt_input=box, prompt_type='box', device=device)
#mask_prompt = format_prompt(n_shots=1, prompt_input=mask, prompt_type='mask', device=device)

with torch.no_grad():
    out = model(video, point_prompt) # choose one between [point_prompt, box_prompt, mask_prompt]
pred_mask = out["pred_masks"][1].sigmoid() > 0.5
```
---
## 2. Paper Results & Training 📘  
_Reproduce benchmarks (strict few-shot & in-context segmentation) and training._
## 📊 Data Preparation
To **[train](#-training)** and **[reproduce our results](#-reproduce-our-results)**, set up your ```dataset```: please refer to [data.md](docs/data.md) for detailed data preparation.      
Once organized, the directory structure should look like this:
```
SANSA/
├── data/
│   ├── COCO2014/
│   ├── FSS-1000/
│   ├── ...
├── datasets/
├── models/
│   ├── sam2/
│   ├── sansa/
│   ├── ...
...
```
---

## 💻 Reproduce our Results

> **· Purpose.** Exact checkpoints and commands to match the paper numbers.  
> **· Tracks.** (1) Strict few-shot segmentation · (2) Generalist in-context segmentation.  
> **· Note.** Models in this section supports masks prompts-only, to ensure fair comparison with prior works.  
> **· Tip.** If you just want one versatile and promptable model for your own data, use [**SANSA Universal Model**](#1-sansa-universal-model-) above.
> 

### (1) Strict Few-Shot Segmentation
Standard **novel-class** protocol with **disjoint partitions**: **LVIS-92<sup>i</sup>** (10 folds) and **COCO-20<sup>i</sup>** (4 folds); **FSS-1000** has a single fixed split.
We release **one adapter per fold** and report **per-fold** and **mean** IoU. Choose shots at eval with `--shot {1|5}`.
Reference objects are given as **masks**.

|        Dataset         |    Pretrained <br/>adapters     | Fold<br/>0 | Fold<br/>1 | Fold<br/>2 | Fold<br/>3 | Fold<br/>4 | Fold<br/>5 | Fold<br/>6 | Fold<br/>7 | Fold<br/>8 | Fold<br/>9 | **Mean<br/>IoU** |
|:----------------------:|:-------------------------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------------:|
|**LVIS-92<sup>i</sup>** | [📥 LVIS (10)](https://drive.google.com/file/d/1Ym_jcIfvAwQUOG-9jiIH_B8iAyJT6Q0L/) |    48.4    |    48.3    |    51.5    |    50.7    |    44.8    |    50.1    |    51.1    |    50.5    |    45.9    |    46.3    |     **48.8**     |
|      **COCO-20<sup>i</sup>**       | [📥 COCO (4)](https://drive.google.com/file/d/1OEWDRqYhRgrXO5RfTuwk_MnnQ8mJApbW/)  |    58.9    |    62.6    |    61.5    |    58.0    |            |            |            |            |            |            |     **60.2**     |
|      **FSS-1000**      |  [📥 FSS-1000](https://drive.google.com/file/d/1Y_W3cL-qxK-J5-yCJk1fXBwNwrhod5RS/)  |    91.4    |            |            |            |            |            |            |            |            |            |     **91.4**     |


**Command to replicate the results:**
```
python inference_fss.py \
  --dataset_file {coco|lvis|fss} \
  --fold {FOLD} \                    # omit for FSS
  --resume /path/to/adapter_{ds}_fold{FOLD}.pth \
  --name_exp eval_{coco|lvis|fss} \
  --shot {1|5} \
  --adaptformer_stages 2,3 \
  --prompt mask
```
*Optionally*, add `--visualize` to visualize the results.


### (2) Generalist In-Context Segmentation
Single **generalist** adapter trained on **COCO+ADE20K+LVIS+PACO** for **in-context few-shot segmentation**: one model across datasets and tasks (**object** + **part** segmentation). 
Reference objects are given as **masks**.

Note: if you want a single generalist **promptable** model,  please refer to [**SANSA Universal Model**](#1-sansa-universal-model-).


|                                          **Pretrained adapters**                                          |    **Segmentation**     |    **Segmentation**     | **Segmentation** |    **Part**     |   **Part**    |
|:---------------------------------------------------------------------------------------------------------:|:-----------------------:|:-----------------------:|:----------------:|:---------------:|:-------------:|
|                                                                                                           | **LVIS-92<sup>i</sup>** | **COCO-20<sup>i</sup>** |   **FSS-1000**   | **Pascal-Part** | **PACO-Part** |
|      [📥 In-Context Generalist](https://drive.google.com/file/d/1ks88lD2exj6GhtCVEK6aatvd1Pq2BBZq/)       |        **50.3**         |        **75.6**         |     **90.0**     |    **49.1**     |   **43.0**    |


**Command to replicate the results:**
```
python inference_fss.py \
  --dataset_file {coco|lvis|fss|pascal_part|paco_part} \
  --fold {FOLD} \                    # LVIS: 0–9, COCO: 0–3, FSS: omit/0, Pascal/PACO: 0–3
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_generalist_fss_{coco|lvis|fss|pascal_part|paco_part} \
  --shot {1|5} \
  --channel_factor 0.8 \
  --adaptformer_stages 2,3 \
  --prompt mask
```

---

## 📈️ Training
### Strict few-shot segmentation

To train SANSA on **strict few shot segmentation**, use the generic command below and adjust the flags as needed:

```
python main.py \
  --batch_size 32 \                 # global batch size (tune to your GPU memory)
  --name_exp train_{ds}_f{FOLD} \   # run name
  --dataset_file {coco|lvis|fss} \  # choose the benchmark
  --fold {FOLD} \                   # fold to EVALUATE on; training uses the REMAINING folds
  --adaptformer_stages 2 3 \        # adapters in the last two Hiera encoder stages
  --prompt mask
```
**Notes:**

- **Strict few-shot protocol:** passing `--fold F` means **evaluate on fold F** and **train on the other folds**.
- **Folds:** COCO-20<sup>i</sup> `F ∈ {0,1,2,3}` · LVIS-92<sup>i</sup> `F ∈ {0,…,9}` · **FSS-1000:** fixed split: omit `--fold`.
- Use `--prompt multi` for **promptable strict few shot segmentation**: trains by sampling among `mask/scribble/box/point` each episode.
- **Frozen SAM2-Large:** backbone/decoder remain frozen; only the adapter is trained.


**Example:**
```
# COCO-20i, fold 0 (strict few-shot)
python main.py --batch_size 32 --name_exp train_coco_f0 --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask
```


---

### Generalist (multi-dataset) training
Train one adapter jointly on multiple datasets:
```
python main.py \
  --batch_size 32 \
  --name_exp train_generalist \
  --multi_train \
  --dataset_file lvis, coco, ade20k, paco_part \
  --ds_weight 0.4, 0.45, 0.1, 0.05 \
  --fold -1 \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --prompt mask
```

**Notes:**
- **`--fold -1` disables strict fold splitting:** for multi-dataset training we don’t use disjoint train/test folds (as we do in strict FSS, where the goal is to evaluate generalization on unseen categories).
- `--ds_weight` sets **per-dataset sampling proportions** (same order as `--dataset_file`).
- To replicate our [**SANSA Universal Model**](#1-sansa-universal-model-), simply add `--prompt multi`.


---

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:


```
@misc{cuttano2025sansa,
      title     = {SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation}, 
      author    = {Claudia Cuttano and Gabriele Trivigno and Giuseppe Averta and Carlo Masone},
      year      = {2025},
      eprint    = {2505.21795},
      url       = {https://arxiv.org/abs/2505.21795}, 
}
```

## Acknowledgements
This project builds upon code from the following libraries and repositories:

- [Segment Anything 2](https://github.com/facebookresearch/sam2)
- [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer)

