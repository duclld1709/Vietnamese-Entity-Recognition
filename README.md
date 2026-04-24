---
title: Vietnamese NER Demo
emoji: 🧠
colorFrom: indigo
colorTo: yellow
sdk: streamlit
sdk_version: 1.46.1
app_file: src/app.py
pinned: false
---
# Vietnamese Named Entity Recognition (NER) 🧠

A comprehensive Vietnamese Named Entity Recognition system using state-of-the-art deep learning models including PhoBERT, CRF, and ensemble methods.


## 🚀 Live Demo

Try the interactive demo: **[Vietnamese NER Demo](https://huggingface.co/spaces/DucLai/Vietnamese_NER)**
It may take a minute for the server to wake up.

![image](https://github.com/user-attachments/assets/ac6f0b96-52f2-4e47-b542-e908c02261c4)

## 🔄 Project Workflow

![Project Flowchart](https://github.com/user-attachments/assets/5b800180-d6c8-44f7-8622-ba188f6cd7be)

## 🎯 Overview

This project implements a robust Vietnamese Named Entity Recognition system that can identify and classify entities in Vietnamese text. The system combines multiple approaches including:

- **PhoBERT-based embeddings** for contextual understanding
- **Conditional Random Fields (CRF)** for sequence labeling
- **Random Forest** with semantic embeddings
- **Rule-based methods** for enhanced accuracy

## 📂 Project Structure

```
VIETNAMESE_NER/
│
├── .github/workflows              
│   └── main.yml                   # Auto deploy to Hugging Space
│
├── data/                          # Dataset files
│   └── raw_data.csv               # Raw training data
│
├── notebooks/                      # Jupyter notebooks for experimentation
│   ├── Duc_Notebook.ipynb         # CRF + RandomForest experiments
│   ├── Softmax_PhoBERT.ipynb      # Softmax approach
│   ├── Kien_Rule_base.ipynb       # Rule-based method with RF
│   └── Kien_RF_lightgbm.ipynb     # RF with semantic embeddings
│
├── src/                           # Main source code
│   ├── __init__.py
│   ├── app.py                     # Streamlit web application
│   ├── front.py                   # Highlight function
│   ├── config.py                  # Project configuration
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── model.py                   # Model architecture definitions
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Model evaluation
│   └── predict.py                 # Inference utilities
│
├── models/                        # Saved model artifacts
│   └── best_model.pt              # Best trained model weights
│
├── outputs/                       # Training outputs
│   ├── output.log                 # Training logs (TensorBoard)
│   └── figures/                   # Visualization plots
│
├── tests/                         # Unit tests (planned)
│
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment file
├── README.md                      # Project documentation
└── run.py                        # Main training script
```


## 🏗️ Model Architecture

The system uses a hybrid architecture combining the strengths of different approaches:

![Model Architecture](https://github.com/user-attachments/assets/82d243a2-42fa-4dad-b1af-8946767d4f44)

### Core Components:
- **PhoBERT-Base**: Generates contextual embeddings for Vietnamese text
- **Linear + CRF Layer**: Handles sequence labeling with context awareness
- **Softmax/Random Forest**: Provides single-label prediction capabilities

## 📊 Dataset & Performance

### Dataset: VLSP2016
The model is trained on the VLSP2016 dataset extracted from Vietnamese news articles.

#### Dataset Statistics:
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/20116929-1556-44b2-86e9-086b72320f22" alt="Entity Frequency" width="600"/></td>
    <td><img src="https://github.com/user-attachments/assets/9cafb068-bbda-4ee1-9fc9-bd4edded1438" alt="Entity Distribution" width="600"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/db9421c0-4e9c-4654-92d0-d924932384dc" alt="Token Length Distribution" width="600"/></td>
    <td><img src="https://github.com/user-attachments/assets/70871bc5-ccb4-4186-9538-ac479c771415" alt="Sentence Length Distribution" width="600"/></td>
  </tr>
</table>


### Model Performance:
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/9fb24f3a-466c-46f1-94d2-bcb6f26abd72" alt="F1 Score" width="600"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/11b8080a-38d6-4ea2-b350-21361345fd1e" alt="Training Loss" width="600"/>
    </td>
  </tr>
</table>

![Results Comparison](https://github.com/user-attachments/assets/e2fecc2c-8b27-4f28-a174-41078b17567c)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Conda (recommended)

### Option 1: Using `requirements.txt`
```bash
# Create and activate conda environment
conda create --name vnner python=3.10
conda activate vnner

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using `environment.yml`
```bash
# Create environment from yml file
conda env create -f environment.yml
conda activate vnner
```

## 🚀 Quick Start

### Training the Model
```bash
python run.py
```

### Running the Streamlit App
```bash
python src/app.py
```

## 🧪 Experimental Approaches

The project explores multiple methodologies:

1. **PhoBERT + CRF**: Sequential labeling with contextual embeddings
2. **PhoBERT + Softmax**: Direct classification approach
3. **Random Forest + Rule-based**: Traditional ML with linguistic rules
4. **Random Forest + Semantic Embeddings**: Enhanced feature engineering

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- VLSP2016 dataset providers
- PhoBERT model creators
- Hugging Face for hosting the demo

## 📚 Additional Resources

For better understanding of the project structure and technologies used:

- [Understanding `__init__.py`](https://zetcode.com/python/init-file/)
- [Markdown Basic Syntax](https://www.markdownguide.org/basic-syntax/#escaping-characters)
- [Requirements.txt vs Environment.yml](https://www.reddit.com/r/learnpython/comments/xvlpdz/why_do_people_provide_a_requirementstxt_or/)

---

**Happy NER-ing! 🎯**
