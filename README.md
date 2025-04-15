# 🍽️ Diverse and High-Quality Food Image Generation Only from Food Name

🎉 **This work was accepted in [ACM TOMM 2025](https://dl.acm.org/journal/tomm)!**

---

## 🧾 Paper Information


<h3 align="center">Diverse and High-Quality Food Image Generation Only from Food Name</h3>

<p align="center">
  <strong>Authors:</strong> Dongjian Yu<sup>1</sup>, Weiqing Min<sup>2</sup>, Xin Jin<sup>1</sup>, Qian Jiang<sup>1</sup>, Ying Jin<sup>2</sup>, Shuqiang Jiang<sup>2</sup><br>
  <sup>1</sup>Yunnan University<br>
  <sup>2</sup>Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences
</p>

[[📄 Paper Link (Coming Soon)]]()  

---

## ✨ Highlights

- 🥘 **Text-to-Image Food Generation**: Our model generates high-quality and diverse food images from **food names only**.
- 🧠 **Multi-modal Feature Fusion**: Combines **common features** (shared semantics) and **private features** (category-specific nuances).
- 📊 **Strong Performance** on four benchmarks: **Food-101**, **VIREO Food-172**, **ISIA Food-200**, and **Food2K**.
- ✅ Evaluated with standard metrics: **Inception Score (IS)**, **Fréchet Inception Distance (FID)**, and **CLIP-I**.
- 🧪 Human evaluation confirms our method produces **more realistic, detailed, and diverse** food images than existing baselines.
- 📚 Ablation studies validate the impact of common/private features, and show that combining both gives the best results.

---

## ⚙️ Setup & Usage

### 1️⃣ Environment Setup

```bash
pip install diffusers==0.23.1
pip install open_clip_torch
pip install torchvision
```
### ▶️ Run
```bash
python cw_food101.py
```
## 📬 Contact
For questions or collaboration:

✉️ Dongjian Yu: yudongjian@ynu.edu.cn

## 🙏 Acknowledgements
Thanks to the open-source community for the great works behind Stable Diffusion, LoRA, CLIP, DisenBooth, etc.

## 📚 Reference

@article{<br>
    title={Diverse and High-Quality Food Image Generation Only from Food Name},<br>
    author={Dongjian Yu and Weiqing Min and Xin Jin and Qian Jiang and Ying Jin and Shuqiang Jiang},<br>
    journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},<br>
    year={2025}<br>
}

