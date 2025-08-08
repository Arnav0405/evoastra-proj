# Image Captioning with InceptionV3 and Attention

This project implements an advanced image captioning model using a deep learning approach. The model utilizes an **encoder-decoder architecture** with an attention mechanism to generate descriptive captions for images.

---

## Project Overview

The core of this project is a sophisticated neural network that understands both images and text.
This project was meant to be a team project between 10 members.

- **Encoder:** We use a pre-trained **InceptionV3** convolutional neural network (CNN) as our encoder. Its role is to extract a grid of rich, spatial features from an input image.
- **Decoder:** An **LSTM**-based recurrent neural network (RNN) serves as the decoder. It takes the image features from the encoder and generates a caption one word at a time.
- **Attention Mechanism:** A critical component that allows the decoder to "look" at the most relevant parts of the image as it generates each word. This dynamic focus helps produce more accurate and contextually relevant captions.
- **Teacher Forcing:** During training, we use this technique to stabilize the learning process by providing the correct word from the training data as input for the next step, rather than the model's own prediction.

---

## Installation

To run this project, you'll need to set up a Python environment with the necessary libraries.

### Prerequisites

- Python 3.x
- TensorFlow 2.x

### Steps

1.  **Clone the repository:**

    ```bash
    git clone
    cd evoastra-proj
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

    _(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing your libraries.)_

---

## Dataset

This model was trained on a specific image captioning dataset. We recommend using one of the following:

- **MS-COCO Dataset:** A large-scale object detection, segmentation, and captioning dataset.
- **Flickr8k/Flickr30k:** Smaller, more manageable datasets ideal for initial experimentation.

You will need to download and preprocess the dataset, including tokenizing the captions and creating a vocabulary.

---

## Usage

### 1\. Pre-process the Data

Before training, you must preprocess your images and captions. The script will handle image resizing and caption tokenization.

```bash
python ./data_extract_code/cap_padding.py
```

### 2\. Train the Model

Check the final-team-c.ipynb file for further details.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
