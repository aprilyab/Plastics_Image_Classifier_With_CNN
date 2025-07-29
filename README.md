#  Intelligent Plastic Classification for Recycling  
##  Building a CNN-Powered Model to Automate Plastic Type Detection

 Date: 16 July – 22 July 2025

##  Business Objective

Plastic waste is one of the most urgent environmental problems of our time. Recycling is a key component of the circular economy, but plastic separation remains expensive and error-prone.

This project develops a deep learning-based image classifier to automate plastic identification — reducing human error and enabling smarter, faster sorting processes in recycling facilities.

---

##  Motivation

- Different plastic types (PET, HDPE, PP, PS) have different properties and must be recycled separately.
- Manual sorting is time-consuming and inconsistent.
- Automating classification helps:
  - Reduce sorting costs
  - Improve recycling efficiency
  - Support sustainable production systems (SDG 12)

Using Convolutional Neural Networks (CNNs) and optionally pre-trained models (e.g., ResNet50), this model identifies plastic types from images and lays the foundation for computer vision-enabled recycling lines.

---

##  Data

The dataset contains images of 4 plastic categories:

- PET (Polyethylene Terephthalate)  
- HDPE (High-Density Polyethylene)  
- PP (Polypropylene)  
- PS (Polystyrene)  

Directory Structure:
Plastics_Image_Classifier_With_CNN/
├──  data/
│   └──  Dataset_plastics_mhadlekar_UN/
│       ├──  HDPE/
│       ├──  LDPA/
│       ├──  PET/
│       ├──  PP/
│       └──  .DS_Store
│
├──  model/
│
├──  notebook/
│   └──  Plastics_Image_Classifier.ipynb
│
├──  venv/                    Virtual environment directory
│
├──  README.md
├──  requirements.txt
└──  .gitignore


Each subfolder contains labeled images of that plastic type.

##  Learning Outcomes

- Perform image preprocessing, augmentation, and normalization
- Build a CNN model from scratch or use a transfer learning approach (ResNet50, VGG16)
- Apply model evaluation using accuracy, precision, and confusion matrix
- Understand the role of deep learning in environmental sustainability
- Use tools like TensorFlow/Keras, Matplotlib, and Scikit-learn


##  Deliverables and Tasks

###  Task 1: Data Exploration & Preprocessing

- Inspect image dimensions and distributions
- Resize images to 256×256, normalize pixel values
- Augment data (rotation, flipping, etc.)
- Split into training (75%) and validation (25%) sets

Deliverables:
- Jupyter notebook with preprocessing pipeline
- Clean, augmented dataset


###  Task 2: Model Building & Training

- Option 1: Custom CNN architecture
- Option 2: Transfer learning using ResNet50
- Train using 30 epochs and appropriate batch size
- Use categorical crossentropy and precision as metrics

Deliverables:
- Trained model saved in model/
- Training and validation accuracy/precision plots


###  Task 3: Evaluation & Visualization

- Plot accuracy and precision across epochs
- Generate a confusion matrix to evaluate per-class performance

Deliverables:
- Matplotlib plots of metrics
- Confusion matrix visualized using sklearn


##  Tools & Libraries

- TensorFlow / Keras  
- Matplotlib, NumPy  
- Scikit-learn  
- Google Colab


---

##  Author

-  Name: Henok Yoseph  
-  Email: henokapril@gmail.com  
-  GitHub: https://github.com/aprilyab/Plastics_Image_Classifier_With_CNN 

---

##  References

- Bakti, A. N., & Shabrina, N. H. (2024). *Plastic Waste Identification using ResNet-50*  
- Mhadlekar et al. (2022). *Plastic Detection using Deep Learning*  
- [TensorFlow ResNet50 Docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
