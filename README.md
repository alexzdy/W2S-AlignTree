# W2S-AlignTree

Implementation for "W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search".


- The [`controlled_sentiment_generation`] directory contains code for using DPO and SFT gpt2 models (124M) to control larger models to write positive movie reviews.


## Installation

```bash
conda create -n w2s_aligntree python=3.10
conda activate w2s_aligntree
pip install -r requirements.txt
```

## Quick Run

```bash
cd ./controlled_sentiment_generation
python w2s_aligntree.py
```
