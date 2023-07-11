# Pizza or Not Pizza Classification Model

This project contains a machine learning model that can classify images as either containing a pizza or not containing a pizza. The model is trained using a deep learning approach and achieves high accuracy in distinguishing between the two classes.

## Dataset

The dataset used for training and evaluation consists of 2,000 images, with 1,000 pizza images and 1,000 non-pizza images. The dataset is carefully curated to ensure a balanced representation of both classes.

## Model Architecture

The model architecture is based on a convolutional neural network (CNN), which is a widely used deep learning architecture for image classification tasks. The CNN consists of multiple convolutional and pooling layers, followed by fully connected layers to perform the final classification.

## Training Process

The model is trained using the Adam optimizer and the binary cross-entropy loss function. The training process is conducted for 30 epochs, with a batch size of 32. Data augmentation techniques, such as random rotations and horizontal flips, are applied to augment the training dataset and improve generalization.

## Evaluation Results

After training, the model achieves an accuracy of 95% on the validation set, demonstrating its ability to accurately classify pizza and non-pizza images. The model's performance is also evaluated using precision, recall, and F1-score metrics to provide a comprehensive understanding of its classification performance.

## Usage

To use the trained model, you can load it into your own Python environment and pass an image through the model for prediction. The model will output a probability indicating the likelihood of the image containing a pizza.

## Dependencies

- Python 3.7 or higher
- TensorFlow 2.0 or higher
- Keras 2.3 or higher
- NumPy, Matplotlib, and other necessary libraries

## License

This project is licensed under the [MIT License](LICENSE), allowing you to use and modify the code for your own purposes.
