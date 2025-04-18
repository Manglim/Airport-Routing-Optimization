# ✈️ Airport Routing Optimization using Neural Networks

This project applies deep learning techniques to predict **airport routing behavior** in a logistics supply chain environment. It uses timing-based features from freight transport legs (planned and actual durations) to classify the next airport in the route.

---

## 🚀 Project Overview

The goal is to predict the **departure airport ID (`i1_dep_1_place`)** for a transport leg using machine learning. The dataset includes masked airport codes and detailed timing metrics across multiple legs of cargo movement.

The solution includes:
- 🧹 Data preprocessing & class filtering  
- 📊 EDA (feature correlation, class distribution)  
- ⭐ Feature selection with ANOVA F-scores  
- 🧠 Neural Network built with TensorFlow/Keras  
- 📈 Accuracy & loss visualization  
- 🛫 Synthetic 2D routing graph using NetworkX

---

## 📁 Dataset Structure

The dataset consists of anonymized transport leg records with features like:
- Planned & actual durations for:
  - RCS (Check-in)
  - DEP (Departure)
  - RCF (Arrival)
  - DLV (Delivery)
- Number of hops per leg (`i1_hops`)
- Masked airport IDs (`i1_dep_1_place`, `i1_rcf_1_place`)

---

## 🧠 Model Architecture

```text
Input Layer (Top 10 selected features)
→ Dense(128, relu)
→ Dropout(0.3)
→ Dense(64, relu)
→ Dropout(0.3)
→ Output Layer (softmax over ~100 classes)
