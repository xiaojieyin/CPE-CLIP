# Constrained Prompt Enhancement for Improving Zero-Shot Generalization of Vision-Language Models

✨ Welcome to the official repository for "Constrained Prompt Enhancement for Improving Zero-Shot Generalization of Vision-Language Models". 


## Installation
```bash
# Require pytorch>=1.10
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies 
pip install -r requirements.txt

```

## Data Preparation
Refer to the following guides for setting up datasets:
- Image datasets: [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) and [SuS-X](https://github.com/vishaal27/SuS-X/blob/main/data/DATA.md)
- Video datasets: [Open-VCLIP](https://github.com/wengzejia1/Open-VCLIP)

## Using CPE-CLIP

```bash
# `testsets` is chosen from:
# ['imagenet', 'oxford_flowers', 'dtd', 'oxford_pets', 'stanford_cars', 'ucf101', 'caltech101', 'food101', 'sun397', 'fgvc_aircraft', 'eurosat']

# Evaluate zero-shot performance
bash ./scripts/evaluate.sh [gpu_id] [testset]
```

## Acknowledgements

Our work builds upon [AWT](https://github.com/MCG-NJU/AWT) and [REAL](https://github.com/shubhamprshr27/NeglectedTailsVLM). Thanks for their excellent work and open-source contributions.
