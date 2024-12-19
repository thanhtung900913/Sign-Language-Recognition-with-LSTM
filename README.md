# Sign Language Recognition with LSTM

This repository contains a project to build a sign language recognition model using LSTM (Long Short-Term Memory networks) on the **WLASL** (Word-Level American Sign Language) dataset. The goal is to recognize individual signs from videos using deep learning techniques.

## Features
- **Dataset**: Preprocessed WLASL dataset for training and evaluation.
- **Model**: LSTM-based model for sequence classification.
- **Libraries**: Utilizes `MediaPipe` for feature extraction and TensorFlow/Keras for model implementation.
- **Training Pipeline**: Includes preprocessing, model training, and evaluation scripts.
- **Visualization**: Displays training metrics and results for better analysis.

## Project Structure
```plaintext
.
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Realtime implementation
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── LICENSE               # License for the project
```

## Dataset
The project uses the **WLASL dataset**, a large-scale dataset for American Sign Language recognition. For more information on WLASL, visit the [official website](https://dxli94.github.io/WLASL/).

### Data Preparation
- Extract keypoints from video frames using `MediaPipe`.
- Normalize and save features into `.npy` files for efficient processing.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sign-language-recognition-lstm.git
   cd sign-language-recognition-lstm
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
- Python 3.7+
- TensorFlow/Keras
- MediaPipe
- NumPy, Pandas, Matplotlib

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [WLASL Dataset](https://dxli94.github.io/WLASL/)
- MediaPipe for efficient feature extraction.
- TensorFlow/Keras for providing robust deep learning tools.

---

### Contact
For questions or suggestions, please open an issue or contact [thanhtung2962004@gmail.com].

