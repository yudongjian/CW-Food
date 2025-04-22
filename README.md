# 🍽️ Diverse and High-Quality Food Image Generation Only from Food Name

🎉 **This work was accepted in [ACM TOMM 2025](https://dl.acm.org/journal/tomm)!**

---

## 🧾 Paper Information


<h3 align="center">Diverse and High-Quality Food Image Generation Only from Food Name</h3>

<p align="center">
   Dongjian Yu<sup>1</sup>, Weiqing Min<sup>2</sup>, Xin Jin<sup>1</sup>, Qian Jiang<sup>1</sup>, Ying Jin<sup>2</sup>, Shuqiang Jiang<sup>2</sup><br>
  <sup>1</sup>Yunnan University<br>
  <sup>2</sup>Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences
</p>

[[📄 Paper Link]](https://doi.org/10.1145/3730588)  

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
type 1:
   python cw_food101.py

type 2:
   train:  self_train_CW_food.py
   generating: ydj_product_image.py
   
   accelerate launch self_train_CW_food.py
      --pretrained_model_name_or_path="/your_path/diffusers_file"
      --instance_data_dir="/your_path/Data/Food2k_complete/42"
      --output_dir="output_dir"
      --instance_prompt="a photo of Home-style_bean_curd"
      --resolution=512
      --train_batch_size=2
      --gradient_accumulation_steps=1
      --checkpointing_steps=200
      --learning_rate=5e-5
      --lr_scheduler="constant"
      --lr_warmup_steps=0
      --max_train_steps=40000
      --validation_prompt="a photo of Home-style_bean_curd"
      --validation_epochs=200
      --seed="0"
      --train_text_encoder
```
## 📬 Contact
For questions or collaboration:

✉️ Dongjian Yu: yudongjian@ynu.edu.cn

## 🙏 Acknowledgements
Thanks to the open-source community for the great works behind Stable Diffusion, LoRA, CLIP, DisenBooth, etc.

## 📚 Reference


```bash
@article{CW-Food,
    title={Diverse and High-Quality Food Image Generation Only from Food Name},
    author={Dongjian Yu and Weiqing Min and Xin Jin and Qian Jiang and Ying Jin and Shuqiang Jiang}
    journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
    year={2025}
    doi = {10.1145/3730588}
}
```
