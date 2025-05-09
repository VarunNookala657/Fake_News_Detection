# Fake News Detection using Machine Learning & NLP

##  Objective
This project aims to classify news articles as **real** or **fake** using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The goal is to help identify misinformation in digital news media.

---

## Technologies Used

| Technology     | Purpose                                      |
|----------------|----------------------------------------------|
| Python         | Core programming language                    |
| Scikit-learn   | Machine learning models and evaluation       |
| Pandas         | Data handling and manipulation               |
| NumPy          | Numerical operations                         |
| NLTK / SpaCy   | Text preprocessing and NLP tasks             |
| Matplotlib / Seaborn | Visualization of data and results    |
| Jupyter Notebook | Interactive experimentation and reporting |

---

##  Folder Structure

- `fake_news_detection/`
  - `data/` – Contains training and test datasets
  - `notebooks/` – Jupyter Notebooks with code and analysis
  - `README.md` – Project documentation

---

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/fake_news_detection.git
    cd fake_news_detection
    ```

2. Create and activate a virtual environment (optional):
    ```bash
    python -m venv venv
    source venv/bin/activate      # Linux/Mac
    venv\Scripts\activate         # Windows
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter notebook to see the model training and evaluation:
    ```bash
    jupyter notebook
    ```

---

##  Models Implemented

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

##  Evaluation

| Model               | Accuracy Score |
|---------------------|----------------|
| Logistic Regression | ~92%           |
| Naive Bayes         | ~89%           |
| SVM                 | **~95%**       |
| Decision Tree       | **~94%**       |
| Random Forest       | ~91%           |

>  **Conclusion**: Based on the results, **SVM** and **Decision Tree** achieved the highest accuracy. These two models are recommended for practical Fake News Detection systems.

---



## Acknowledgements

This project was developed as part of an assignment or personal exploration into NLP and misinformation detection using machine learning.

---

