# 🥔 Potato Leaf Disease Classification

This project aims to classify potato leaf conditions (Early Blight, Late Blight, Healthy) using deep learning models trained on images from the PlantVillage dataset.

## 📌 Models Used
We implemented and compared the performance of three deep learning models:
- ✅ **CNN** (Custom-built): High training accuracy but showed signs of overfitting.
- ✅ **VGG16**: Pretrained on ImageNet, stable performance with moderate generalization.
- ✅ **MobileNet**: Best performance with 99.95% test accuracy and lowest loss. Ideal for deployment on low-resource devices.

## 🗂 Dataset
- **Source**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classes**: Early Blight, Late Blight, Healthy
- **Images**: Augmented to balance class representation.

## 📊 Results Summary

| Model      | Training Acc | Validation Acc | Test Acc | Test Loss |
|------------|--------------|----------------|----------|-----------|
| CNN        | 99.21%       | 95.6%          | 91.66%   | 0.4002    |
| VGG16      | 97.34%       | 97.92%         | 93.75%   | 0.1871    |
| **MobileNet** | **99.7%**   | **99.27%**     | **99.95%** | **0.0537** |

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- Jupyter Notebooks
- Google Colab
- Data Augmentation
- Image Classification

## 📁 Notebooks
- 🧩 [CNN Model Notebook](https://github.com/ALHADDAD98/CPCS433/blob/main/CPCS433_CNN)
- 🔍 [VGG16 Notebook](https://github.com/AlaaEmad1205/AlgorithmProject/blob/main/VGG_16_(2).ipynb)
- ⚡ [MobileNet Notebook](https://github.com/AlaaEmad1205/AlgorithmProject/blob/main/MobileNet_Model.ipynb)


## 📌 Future Work
- Ensemble methods (stacking, bagging)
- Hyperparameter tuning using Grid Search or Bayesian Optimization
- Expand dataset with more real-world images

---

## 💡 About the Team
> Developed as a collaborative project in [CPCS433] Deep Learning Course.

---

