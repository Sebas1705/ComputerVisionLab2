# Practica2VisionArtificial

Computer Vision university lab — OCR character recognition with LDA/PCA + classical ML classifiers.

Extends Lab 1 (sign panel detection) with an OCR pipeline: extract character images from
detected panels, reduce dimensionality with LDA and PCA, then benchmark six sklearn
classifiers (SVC, Random Forest, KNN, Logistic Regression, Decision Tree, Gaussian NB).

---

## Exercises

| Exercise | Script | Description |
|---|---|---|
| Ex1 | `Scripts/Executors/Ex1.py` | LDA → classifier benchmark |
| Ex2 | `Scripts/Executors/Ex2.py` | PCA → classifier benchmark |
| Ex3 | `Scripts/Executors/Ex3.py` | Panel detection pipeline (from Lab 1) |
| Ex4 | `Scripts/Executors/Ex4.py` | Full OCR pipeline end-to-end |

---

## Project Structure

```
Practica2VisionArtificial/
├── proyect/
│   ├── src/
│   │   ├── Classes/
│   │   │   ├── Common/
│   │   │   │   ├── ImageLoader.py           # Abstract base for loading images from directories
│   │   │   │   └── ImagePreproccesor.py     # Grayscale, adaptive threshold, contours (abstract base)
│   │   │   ├── Ex1AndEx2/
│   │   │   │   ├── CharactersLoader.py      # Load OCR character images (train/validation)
│   │   │   │   ├── CharactersPreprocessor.py # Preprocess characters for feature extraction
│   │   │   │   └── ClassifierTester.py      # Train & benchmark sklearn classifiers (threaded)
│   │   │   ├── Ex3/
│   │   │   │   ├── PanelsLoader.py          # Load sign panel images
│   │   │   │   └── PanelsPreprocessor.py    # Preprocess panels
│   │   │   └── OCR/
│   │   │       ├── ocr_classifier.py        # OCR character↔label mapping
│   │   │       └── lda_normal_bayes_classifier.py
│   │   ├── Scripts/
│   │   │   ├── Executors/                   # Ex1.py – Ex4.py entry points
│   │   │   ├── Tests/                       # Test runners
│   │   │   └── OCR/
│   │   │       ├── evaluate_ocr_classifiers.py     # Standalone OCR evaluation script
│   │   │       └── evaluate_ocr_panels_results.py  # Panel OCR result evaluation
│   │   ├── Common/
│   │   │   ├── Settings.py                  # Paths and global constants
│   │   │   └── FileFuncs.py                 # Image I/O helpers
│   │   ├── p1/                              # Lab 1 pipeline (integrated as dependency)
│   │   └── main.py                          # Entry point — runs all exercises
│   ├── images/                              # Input and intermediate images
│   └── files/
│       ├── LDA/                             # Classification reports per classifier (LDA)
│       └── PCA/                             # Classification reports per classifier (PCA)
├── aux_dir/
│   └── Practica_2_DCP.ipynb               # Jupyter exploration notebook
└── README.md
```

---

## Running

```bash
cd proyect/src
python main.py
```

Runs all four exercises sequentially:
1. Ex1 — LDA dimensionality reduction + classifier benchmarks
2. Ex2 — PCA dimensionality reduction + classifier benchmarks
3. Ex3 — Panel detection
4. Ex4 — Full end-to-end OCR

---

## Classifier Results

Results are saved as text files in `proyect/files/LDA/` and `proyect/files/PCA/`.
Each file contains the sklearn `classification_report` and execution time for one classifier.

---

## Tech Stack

- Python 3.12
- OpenCV (`cv2`) — image preprocessing
- scikit-learn — LDA, PCA, SVC, Random Forest, KNN, Logistic Regression, Decision Tree, GaussianNB
- NumPy, Matplotlib
- Threading — classifiers trained in parallel

---

## Course Context

Lab 2 of the *Computer Vision* (Visión Artificial) course.
Goal: build an end-to-end OCR pipeline for road sign panels — from raw image to character
recognition — and compare classical dimensionality reduction techniques (LDA vs PCA)
combined with multiple classifiers.
