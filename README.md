![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# PROJECT - Deep-Learning-Image-classification-using-CNN

### Project Overview:
This project implements **VGG16 transfer learning and fine-tuning** to classify images into 10 categories from the CIFAR-10 dataset. The approach involves leveraging the pre-trained VGG16 model, applying additional custom layers, and fine-tuning to improve performance on the target dataset. The pipeline includes dataset preprocessing, model architecture design, training, evaluation, and visualization.

The focus is solely on VGG16 and demonstrates the application of deep learning in computer vision.

---

### Table of Contents
- Folder Structure
- Environment Setup
- Project Components
- Usage
- Future Enhancements

---

### Folder Structure:
- **`main.ipynb`**: Jupyter Notebook containing the end-to-end pipeline for dataset preparation, model design, and evaluation.
- **README.md**: Project documentation (this file).
- **training_data/ and test_data/**: CIFAR-10 dataset directories (optional for local storage).
- **requirements.txt**: Lists all dependencies required for the project.
- **Group2 - Image Classification with CNN_Report.pdf**: Summary report of the project.
- **Group2 - Image Classification with CNN_Presentation.pdf**: Slides presenting key insights, results, and methodologies.

---

### Project Components:
1. **Dataset Preparation**:
   - CIFAR-10 dataset is loaded, normalized (pixel values scaled to [0, 1]), and one-hot encoded for classification.

2. **Transfer Learning with VGG16**:
   - Utilizes the pre-trained VGG16 model from TensorFlow/Keras.
   - Adds custom fully connected layers for CIFAR-10 classification.
   - Fine-tunes VGG16 layers to adapt to the target dataset.

3. **Model Training**:
   - Includes callbacks like `EarlyStopping` and `ModelCheckpoint`.
   - Optimized with Adam optimizer and categorical cross-entropy loss.

4. **Evaluation and Visualization**:
   - Performance metrics include accuracy, confusion matrix, and loss plots.
   - Visualizations of predictions and misclassifications.

---

### Usage:
1. **Run the Jupyter Notebook**:
   - Open `main.ipynb` and execute the cells in sequence.

2. **Dataset Loading**:
   - CIFAR-10 dataset is automatically downloaded via TensorFlow's API.

3. **Model Training and Evaluation**:
   - Train the model using the provided architecture and visualize results.

---

### Future Enhancements:
- Explore other architectures like ResNet50 and EfficientNet for comparison.


