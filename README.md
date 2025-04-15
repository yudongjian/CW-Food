# Diverse and High-Quality Food Image Generation Only from Food Name

🎉 **This work was accepted in ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 2025!**

## 📝 Paper Information

**Title:** Diverse and High-Quality Food Image Generation Only from Food Name  
**Authors:**  
Dongjian Yu<sup>1</sup>, Weiqing Min<sup>2</sup>, Xin Jin<sup>1</sup>, Qian Jiang<sup>1</sup>, Shaowen Yao<sup>1</sup>, Shuqiang Jiang<sup>2</sup>  
<sup>1</sup>Yunnan University  
<sup>2</sup>Key Laboratory of Intelligent Information Processing,  
Institute of Computing Technology, Chinese Academy of Sciences

[[📄 Paper Link (Coming Soon)]]()  
[[📷 Project Page (Optional)]]()

---

## 🚀 Highlights

- 🔤 **Text-only Input**: Generate realistic food images using only the food name.
- 🔗 **Multi-modal Embedding**: Combines textual and visual semantics via food name embeddings and image priors.
- 🧠 **Pretrained Diffusion Models**: Leverages Stable Diffusion and LoRA to produce diverse and high-quality results.
- 🍜 **Supports Various Cuisines**: Successfully generates both Chinese and Western food images.
- 📊 **Extensive Evaluation**: Evaluated on Food-101, VIREO Food-172, ISIA Food-200, and Food2K datasets with metrics such as IS, FID, and CLIP-I.

---

## 📦 Code (Coming Soon)

> 🚧 The code is being cleaned up and will be released soon. Stay tuned!

---

## 📈 Citation

If you find our work helpful, please cite us:

```bibtex
@article{yu2025diverse,
  title={Diverse and High-Quality Food Image Generation Only from Food Name},
  author={Yu, Dongjian and Min, Weiqing and Jin, Xin and Jiang, Qian and Yao, Shaowen and Jiang, Shuqiang},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  year={2025}
}


Prepare:	

	Install the conda virtual environment:
		pip install diffusers(0.23.1)
		pip install open_clip_torch
		pip install torchvision

	Prepare the Food-101 dataset and stable diffuser file.

	You need to modify the path in cw_food101.py.

run:
	python cw_food101.py
	
	
