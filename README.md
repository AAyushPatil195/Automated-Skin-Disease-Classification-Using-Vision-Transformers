# 🧠 Vision Transformer (ViT) for Skin Disease Classification

This project uses a **Vision Transformer (ViT)** to classify skin disease images into 14 categories using a dataset from Kaggle. It demonstrates how transformer-based models can be applied to computer vision tasks, replacing traditional CNNs with self-attention mechanisms for powerful global feature extraction.

---

## 📁 Dataset

- Dataset: [Skin Disease Dataset (ahmedxc4/skin-ds)](https://www.kaggle.com/datasets/ahmedxc4/skin-ds)
- Automatically downloaded via `kagglehub`.

---

## 🔧 Features

- Vision Transformer (ViT) architecture (`google/vit-base-patch16-224-in21k`)
- Multi-head self-attention for global image understanding
- Class balancing using `WeightedRandomSampler`
- Training & validation accuracy/loss tracking
- Evaluation using classification report and confusion matrix
- GPU support (via CUDA)

---

## 📊 Classes

This model classifies images into **14 skin disease categories**.  
*(Ensure the dataset is organized into `train/`, `val/`, and `test/` folders with subdirectories for each class.)*

---

✅ Evaluation
After training, the model is evaluated on a separate test set using:

- Accuracy

- Classification report

- Confusion matrix (via Seaborn)

🧠 How Attention Works
The model uses the self-attention mechanism from transformers to assign different weights to different patches of an image, focusing more on the relevant areas.

Self-Attention Formula:
  Attention (Q, K, V) = softmax [ ( Q * (K^T) ) / sqrt(d) ] * V

📈 Accuracy Formula
Accuracy = (Correct Predictions / Total Predictions) × 100


🧪 How to Avoid Overfitting
- Used data augmentation and class balancing

- May include dropout, early stopping, or regularization for future improvement

🔍 Activation Functions
The ViT model uses GELU activation functions internally, which work well with transformer architectures.

📌 Future Improvements
- Add model checkpointing and early stopping

- Use more advanced learning rate scheduling

- Deploy the model using a Flask or FastAPI backend

📷 Sample Result

- Confusion matrix of test set performance.

🧠 Inspiration
This project was inspired by the Vision Transformer (ViT) architecture and its success in replacing CNNs with self-attention for image classification tasks.

📚 References
- ViT Paper

- Hugging Face Transformers

- Kaggle Dataset

![image](https://github.com/user-attachments/assets/45aba0e6-2fa2-4cf7-881c-4fe5c6dee8cf)
![image](https://github.com/user-attachments/assets/e5bee6fa-3266-448e-8519-f5be71fb6cae)
![FinalDesign](https://github.com/user-attachments/assets/f217af61-44ee-4d2f-a85b-7d34cc1b2839)
![image](https://github.com/user-attachments/assets/9a699089-a3f1-48ce-b21e-4075d4c84d17)
![image](https://github.com/user-attachments/assets/caedbab2-f537-4ffb-beff-19bd98f5e60b)
