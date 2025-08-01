# Retina

**Official implementation of**  
📄 *"Memory-Efficient and Low-Latency Multi-modal Document Retrieval for NL Queries using MLLMs"*  

---

Retina is a **memory-efficient** and **low-latency** multi-modal document retrieval framework for natural language queries, powered by Multi-modal Large Language Models (MLLMs). 

---

## 🔍 Overview

Retina reconstructs the retrieval stack for multi-modal documents, combining:

- **Sparse MLLM-based filtering** 
- **Efficient late interaction** 
- **Token-level multi-vector representations** 
- **Flexible integration** with visual documents (PDFs, charts, slides, etc.)

---

## 🚀 Highlights

- 🧠 Sparse filtering with fine-tuned MLLMs
- 🧊 Memory-efficient late interaction design
- ⚡ Low-latency query processing for real-time applications

---

## 📂 Repository Structure

```
Retina/
├── LLM4IR/                       # Model implementations
├── scripts/                      # Training and inference scripts
│   └── configs/                  # YAML configs for different settings
├── README.md                    # Project README
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/ZJU-DAILY/Retina.git
cd Retina
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt 
```

### 3. Train a Retriever

```bash
bash scripts/train/train_colbert.sh

###updateing.......
