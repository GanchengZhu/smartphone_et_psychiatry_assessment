# Schizophrenia Dataset Processing for Eye Tracking: Smartphone and EyeLink

This repository provides the full data processing and machine learning analysis pipeline for schizophrenia-related eye movement datasets, collected from both **smartphone-based eye tracking** and **EyeLink systems**.

Please request data at  https://www.dlab.org/data-set#mental-health
and decompress them into the following folders:

```bash
schizophrenia_detection/data_eyelink_asc
schizophrenia_detection/data_phone
schizophrenia_detection/meta_data
depression_symptom_detection/ground_truth
depression_symptom_detection/raw_data
```

---

## 📱 Smartphone Eye-Tracking Pipeline

Run the following commands, please first run this: `cd schizophrenia_detection`.

### 1. Detect Eye Movement Events

Run the following commands to detect fixations and saccades from raw smartphone-based eye-tracking data:

```bash
python phone_eye_events_detection.py --raw_data_path batch_0
python phone_eye_events_detection.py --raw_data_path batch_1
python phone_eye_events_detection.py --raw_data_path batch_1_sz_repeat_measure
```

---

### 2. Extract Eye Movement Features

Use the script below to extract temporal and spatial features from the detected events:

```bash
python phone_extract_features.py --raw_data_path batch_0
python phone_extract_features.py --raw_data_path batch_1
python phone_extract_features.py --raw_data_path batch_1_sz_repeat_measure
```

---

### 3. Machine Learning Analysis

#### Model Training

```bash
python train.py --classifier svm --data_source phone
python train.py --classifier lr --data_source phone
python train.py --classifier knn --data_source phone
python train.py --classifier nb --data_source phone
python train.py --classifier dtree --data_source phone
python train.py --classifier rf --data_source phone
python train.py --classifier bagging --data_source phone
python train.py --classifier catboost --data_source phone
```

#### Model Testing

```bash
python test.py --classifier svm --data_source phone
python test.py --classifier lr --data_source phone
python test.py --classifier knn --data_source phone
python test.py --classifier nb --data_source phone
python test.py --classifier dtree --data_source phone
python test.py --classifier rf --data_source phone
python test.py --classifier bagging --data_source phone
python test.py --classifier catboost --data_source phone
```

---

## 🎯 EyeLink Eye-Tracking Pipeline


### 1. Detect Eye Movement Events

```bash
python eyelink_eye_events_detection.py --raw_data_path batch_0
python eyelink_eye_events_detection.py --raw_data_path batch_1
python eyelink_eye_events_detection.py --raw_data_path batch_1_sz_repeat_measure
```

---

### 2. Extract Features from `.asc` Files

```bash
python eyelink_extract_features.py --raw_data_path batch_0
python eyelink_extract_features.py --raw_data_path batch_1
python eyelink_extract_features.py --raw_data_path batch_1_sz_repeat_measure
```

---

### 3. Machine Learning Analysis

#### Model Training

```bash
python train.py --classifier svm --data_source eyelink
python train.py --classifier lr --data_source eyelink
python train.py --classifier knn --data_source eyelink
python train.py --classifier nb --data_source eyelink
python train.py --classifier dtree --data_source eyelink
python train.py --classifier rf --data_source eyelink
python train.py --classifier bagging --data_source eyelink
python train.py --classifier catboost --data_source eyelink
```

#### Model Testing

```bash
python test.py --classifier svm --data_source eyelink
python test.py --classifier lr --data_source eyelink
python test.py --classifier knn --data_source eyelink
python test.py --classifier nb --data_source eyelink
python test.py --classifier dtree --data_source eyelink
python test.py --classifier rf --data_source eyelink
python test.py --classifier bagging --data_source eyelink
python test.py --classifier catboost --data_source eyelink
```

---


# Depression Dataset Processing for Eye Tracking

Run the following commands, please first run this: `cd depression_symptom_detection`.

## Pipeline

### 1. Detect Eye Movement Events

```bash
python phone_eye_events_detection.py
```

---

### 2. Extract Features 

```bash
python extract_features.py
```

---

### 3. Model training

```bash
python train.py --classifier svm
python train.py --classifier lr
python train.py --classifier knn
python train.py --classifier nb
python train.py --classifier dtree
python train.py --classifier bagging
python train.py --classifier rf
python train.py --classifier catboost
python test.py --classifier svm
python test.py --classifier lr
python test.py --classifier knn
python test.py --classifier nb
python test.py --classifier dtree
python test.py --classifier bagging
python test.py --classifier rf
python test.py --classifier catboost



```

---

### Model testing

```bash
python test.py --classifier svm
python test.py --classifier lr
python test.py --classifier knn
python test.py --classifier nb
python test.py --classifier dtree
python test.py --classifier bagging
python test.py --classifier rf
python test.py --classifier catboost


```

---
