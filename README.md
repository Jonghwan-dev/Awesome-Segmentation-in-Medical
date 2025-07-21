---

# **Awesome Segmentation in Medical Imaging: A Breast Ultrasound Benchmark 🚀**

A robust and reproducible framework for benchmarking deep learning models on Breast Ultrasound (BUS) image segmentation.1

The core philosophy of this project is to establish a **fair and unbiased evaluation pipeline**, addressing common pitfalls in machine learning research like test set leakage. By strictly separating cross-validation from final testing, this framework produces publication-ready results that accurately reflect a model's true generalization performance.

*(An example GIF demonstrating segmentation results would be ideal here.)*

## **Table of Contents**

- [Our Vision](#our-vision-)
- [Key Features](#key-features-)
- [Supported Models & Datasets](#supported-models--datasets-)
- [Project Structure](#project-structure-)
- [Installation](#installation-)
- [Running Experiments](#running-experiments-)
- [Understanding the Results](#understanding-the-results-)
- [How to Extend](#how-to-extend-️)
- [Acknowledgements](#acknowledgements-)
- [License & Citation](#license--citation-)

---

## **Our Vision 💡**

While research in medical image segmentation is advancing rapidly, fair model comparison remains a significant challenge due to:

1. **Data Leakage:** The test set is often inadvertently used during cross-validation, leading to inflated and unreliable performance metrics.  
  
2. **Inconsistent Evaluation:** Different studies use different data splits, preprocessing steps, and evaluation metrics, making direct comparisons impossible.  
  
3. **Limited Scope:** Most benchmarks focus only on binary (benign vs. malignant) classification, failing to incorporate the 'normal' class, which is crucial for real-world clinical applications.
  
  
Awesome-BUS-Benchmark is engineered to solve these problems. We provide a standardized benchmark built on the principles of **strict data separation** and **stratified sampling** to ensure that all models are evaluated under the exact same conditions, leading to truly comparable and reproducible results.

---

## **Key Features ✨**

* **🥇 Strict Data Splitting:** A dedicated, held-out test set is created **once** before any training or cross-validation begins. This test set is never seen during model development or selection, preventing any form of data snooping and ensuring a truly unbiased final evaluation of generalization performance.  
  
* **⚖️ Stratified K-Fold Cross-Validation:** Implements Stratified K-Fold to handle the inherent class imbalance in BUS datasets (e.g., the small number of normal cases). This ensures that each fold's class distribution is representative of the overall dataset, leading to more stable training and reliable validation metrics.  
  
* **📚 Comprehensive Dataset Support:** Natively supports multiple public Breast US datasets and crucially **includes the normal class**, offering a more complete and realistic benchmark than typical binary (benign vs. malignant) studies.  
  
* **🧩 Modular & Extensible Architecture:** The code is structured to be highly modular. You can easily add new datasets, models (both CNN & Transformer-based), loss functions, and metrics with minimal code changes.  
  
* **⚙️ Automated & Configurable Pipelines:** Comes with powerful shell scripts (run\_cnn.sh, run\_vit.sh) that automate the entire workflow: k-fold training, testing, and results aggregation. All experiment parameters (model choice, learning rate, epochs, etc.) are controlled via central YAML configuration files, allowing for rapid and reproducible experiments.
  
  
---
  

## **Supported Models & Datasets 📖**

### **Model Zoo**
|     Model     |                        Original code                         |                          Reference                           |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      **U-Net**      | [Caffe](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net) |      [MICCAI'15](https://arxiv.org/pdf/1505.04597.pdf)       |
| **Attention U-Net** | [Pytorch](https://github.com/ozan-oktay/Attention-Gated-Networks) |       [MIDL'18](https://arxiv.org/pdf/1804.03999.pdf)       |
|     **U-Net++**     |    [Pytorch](https://github.com/MrGiovanni/UNetPlusPlus)     | [MICCAI'18](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7329239/pdf/nihms-1600717.pdf) |
|    **U-Net 3+**     |    [Pytorch](https://github.com/ZJUGiveLab/UNet-Version)     |        [ICASSP'20](https://arxiv.org/pdf/2004.08790)         |
|    **TransUnet**    |      [Pytorch](https://github.com/Beckschen/TransUNet)       |       [Arxiv'21](https://arxiv.org/pdf/2102.04306.pdf)       |
|      **MedT**       | [Pytorch](https://github.com/jeya-maria-jose/Medical-Transformer) |      [MICCAI'21](https://arxiv.org/pdf/2102.10662.pdf)       |
|      **UNeXt**     | [Pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch)  |      [MICCAI'22](https://arxiv.org/pdf/2203.04967.pdf)       |
|    **SwinUnet**     |    [Pytorch](https://github.com/HuCaoFighting/Swin-Unet)     |       [ECCV'22](https://arxiv.org/pdf/2105.05537.pdf)        |
|     **CMU-Net**     |       [Pytorch](https://github.com/FengheTan9/CMU-Net)       |       [ISBI'23](https://arxiv.org/pdf/2210.13012.pdf)        |
|     **CMUNeXt**     |       [Pytorch](https://github.com/FengheTan9/CMUNeXt)       |       [ISBI'24](https://arxiv.org/pdf/2308.01239.pdf)       |

### **Datasets**

| Dataset | Official Source | Download link|
| :---- | :---- | :---- |
| **BUSI** | [Dataset of Breast Ultrasound Images \- Al-Dhabyani et al.](https://www.sciencedirect.com/science/article/pii/S2352340919312181?via%3Dihub) | [Download Data](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |
| **BUSBRA** | [Breast Ultrasound Bi-Rads Classification... \- Ribeiro et al.](https://pubmed.ncbi.nlm.nih.gov/37937827/) | [Download Data](https://zenodo.org/records/8231412) |
| **BUS-UC** | [Breast Ultrasound Cancer Image Classification \- Garodia et al.](https://www.sciencedirect.com/science/article/pii/S0952197623014768?via%3Dihub) | [Download Data](https://data.mendeley.com/datasets/3ksd7w7jkx/1) |
| **BUS-UCLM** | [BUS-UCLM: Breast ultrasound lesion segmentation dataset. \- Noelia Vallez et al.,2025](https://www.nature.com/articles/s41597-025-04562-3) | [Download Data](https://github.com/noeliavallez/BUS-UCLM-Dataset) |
| **Yap2018** | [Breast ultrasound lesions recognition: a preliminary study... \- Yap et al.](https://pubmed.ncbi.nlm.nih.gov/28796627/) | [Download Data](https://www2.docm.mmu.ac.uk/STAFF/m.yap/files/BUS_ReleaseAgreement.pdf) |

  
---
  
## **Project Structure 📂**

The repository is organized logically to separate concerns and facilitate ease of use and extension.
```
Awesome_Segmentation_in_Medical/  
│
├── data/  
│   ├── preprocessing/  
│   │   ├── augmentation.py      # Data augmentation logic (rotations, flips, etc.)  
│   │   └── preprocess.py        # Data preprocessing logic (resizing, normalization, etc.)  
│   ├── prepare_datasets.py      # Script to standardize raw datasets and create CSVs with fold splits  
│   └── synthetic_datasets.py    # Script for synthetic data generation (optional)  
│  
├── data_loader/  
│   └── data_loaders.py          # Defines PyTorch DataLoaders for training/validation/testing  
│  
├── datasets/                    # (User-supplied) Directory to store raw downloaded datasets  
│  
├── src/  
│   ├── models/                  # PyTorch model architecture definitions  
│   │   ├── cnn_based/           # --- CNN-based models like UNet, AttUNet, UNet++, UNeXt, CMUNet  
│   │   └── ViT_based/           # --- Transformer-based models like TransUnet, Swin-Unet, MedT  
│   │  
│   ├── trainer/  
│   │   └── trainer.py           # Core training and validation loop logic (epochs, backprop, etc.)  
│   │  
│   └── utils/                   # Core utilities and helper functions  
│       ├── losses.py            # --- Loss functions for segmentation (DiceLoss, BCELoss, etc.)  
│       ├── metrics.py           # --- Evaluation metrics (Dice, IoU, HD95, etc.)  
│       ├── parse_config.py      # --- Functionality to read and parse the config.json file  
│       └── util.py              # --- Other useful functions, such as logging  
│  
├── results/                     # Stores all experiment outputs, including CSVs with metrics per fold  
│  
├── .gitignore                   # List of files to be ignored by Git  
├── config.json                  # Central configuration file to control all experiments (models, hyperparameters, etc.)  
├── environment.yml              # File for Conda environment setup  
│  
├── run_cnn.sh                   # Entrypoint script to run CNN-based model experiments  
├── run_vit.sh                   # Entrypoint script to run Transformer-based model experiments  
├── run_transfer.sh              # Entrypoint script for transfer learning experiments  
│  
├── train.py                     # Main executable file to start model training  
└── test.py                      # Main executable file to test a trained model     
```

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Jonghwan-dev/Awesome_Segmentation_in_Medical.git
   cd Awesome_Segmentation_in_Medical
   ```

2. **Conda (recommended)**

   ```bash
   conda env create -f environment.yml
   conda activate awesome_seg
   ```

3. **Download datasets**

   - Create a `datasets/` folder in the project root
   - For each public dataset (BUSI, BUSBRA, BUS-UC, BUS-UCLM, Yap2018), follow the original authors’ instructions to place images and masks under:
     ```
     datasets/
     ├── BUSI/
     │   ├── images/
     │   └── masks/
     └── BUSBRA/
         ├── images/
         └── masks/
     ```

---

## 🛠️ Data Preparation

Standardize raw data, generate CSV manifests and split files:

```bash
python -c "from data.prepare_datasets import PrepareDataset; PrepareDataset().run(['busi','busbra','bus_uc','bus_uclm','yap'])"
```

This produces:

- `data/csv/` with `busi.csv`, `busbra.csv`, etc.
- `data/splits/` containing one held‑out test split and 5 stratified folds for CV.

---

## 🚀 Running Experiments

### 1. CNN Models

```bash
bash scripts/run_cnn.sh
```

Performs k-fold cross-validation for UNet, AttUNet, UNet++ and UNet3+, then evaluates on the held‑out test set.

### 2. Transformer Models

```bash
bash scripts/run_vit.sh
```

Trains TransUNet, Swin-Unet and MedT with the same splits.

### 3. Transfer Learning

```bash
bash scripts/transfer_run.sh
```

Fine-tunes CNN or Transformer backbones pretrained on natural images.

All logs and per-fold metrics are saved under `results/`.

---

## **Understanding the Results 📊**

The performance tables should be populated with the results generated in the results/ directory. The evaluation methodology is key:

The metrics reported are calculated on the **held-out test set**. The predictions on this test set are an **average of the 5 models** trained during the 5-fold cross-validation. This ensemble approach provides a more robust and stable measure of the model's true generalization capability, reducing the impact of random initialization or fold-specific performance variations.

### **BUSI Dataset Performance**

| Model | Dice (DSC) | IoU | HD95 | GFLOPs | Params (M) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| UNet | \- | \- | \- | 17.34 | 7.76 |
| AttUNet | \- | \- | \- | 18.06 | 8.71 |
| SwinUnet | \- | \- | \- | 13.96 | 27.20 |
| ... | ... | ... | ... | ... | ... |

*(Repeat tables for BUSBRA, BUS-UC, BUS-UCLM, and Yap2018 datasets here.)*

---

## **How to Extend 🛠️**

### **Add a New Model**

1. Place your model's .py file in models/cnn\_models/ or models/transformer\_models/.  
2. Import your model in the corresponding \_\_init\_\_.py file.  
3. Add the new model's name (as a string) to the model\_name list in the relevant configs/\*.yml file.

### **Add a New Dataset**

1. Add the raw dataset folder to the datasets/ directory.  
2. In data/prepare\_datasets.py, add a new preparation method (e.g., \_prepare\_mynew\_dataset) inside the PrepareDataset class.  
3. Register your new method in the dispatcher dictionary within the run method.

---

## **Acknowledgements 🙏**

+ This project's structure and methodology are heavily inspired by [Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks).

+ Helper functions from [CMU-Net](https://github.com/FengheTan9/CMU-Net) and [Image\_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation) were also utilized. We extend our gratitude to the authors of these repositories for making their excellent work public.

---

## License & Citation 📜

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
If you use this benchmark in your research, please consider citing it:
```
@misc{awesomebusbenchmark2024,  
  author \= {Jonghwan Kim},  
  title \= {Awesome Segmentation in Medical Imaging: A Breast Ultrasound Benchmark},  
  year \= {2025},  
  publisher \= {GitHub},  
  journal \= {GitHub repository},  
  howpublished \= {\\url{https://github.com/Jonghwan-dev/Awesome\_Segmentation\_in\_Medical}},  
}
```