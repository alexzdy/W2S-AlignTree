# AAAI 2026 Oral | W2S-AlignTree

Implementation for "W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search".

## 📰 News

🎉 **Accepted as Oral at AAAI 2026.**

- Paper on arxiv: [W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](https://arxiv.org/abs/2511.11518).

## 💡 Highlights
<div align="center">
  <img src="assets/main.png" alt="Image 1" style="width: 80%;">
</div>

- **First integration of MCTS with Weak-to-Strong Generalization:** W2S-AlignTree pioneers a plug-and-play inference-time alignment framework that combines Monte Carlo Tree Search with weak-to-strong guidance, enabling dynamic, fine-grained control over strong LLMs without any parameter updates.

- **Entropy-Aware PUCT for adaptive exploration:** Introduces a novel selection rule that incorporates policy entropy to intelligently balance exploration-exploitation, preventing premature convergence and improving trajectory diversity in complex generation spaces.

- **Plug-and-play deployment with weak-model guidance:** Enables immediate alignment of frozen strong LLMs across diverse tasks and model families, achieving superior performance while significantly reducing computational costs through lightweight weak model proxies.

## 🛠️ Installation

```bash
conda create -n w2s_aligntree python=3.10
conda activate w2s_aligntree
pip install -r requirements.txt
```

## 🚀 Quick Run

- The [`controlled_sentiment_generation`] directory contains code for using DPO and SFT gpt2 models (124M) to control larger models to write positive movie reviews.
- The [`summarization`] directory contains code for using DPO and SFT gpt2 models (124M) to control larger models to generate a summary.

```bash
cd ./controlled_sentiment_generation && ./summarization
python w2s_aligntree.py
```

## 📖 Citation
If you find W2S-AlignTree useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:
```
@misc{ding2025w2saligntreeweaktostronginferencetimealignment,
      title={W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search}, 
      author={Zhenyu Ding and Yuhao Wang and Tengyue Xiao and Haoying Wang and Guojun Ma and Mingyang Wan and Caigui Jiang and Ning Ding},
      year={2025},
      eprint={2511.11518},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.11518}, 
}
```
