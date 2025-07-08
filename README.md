# Low-Cost Urban Spatial Data Extraction for Energy Modeling via Few-Shot Computer Vision

**🚀 [Quick Start - Jump to Workflow](#workflow)**

## Project Overview

This project is part of the Stanford University UGVR (Undergraduate Visiting Research) program, aimed at automatically extracting urban spatial data from publicly available satellite imagery using few-shot computer vision techniques for Urban Building Energy Modeling (UBEM). The core innovation lies in using a pre-trained CLIPSeg vision-language model that can be fine-tuned with only 20 manually labeled images to achieve accurate urban feature segmentation.

### Target Classes
The project identifies 6 urban feature classes:
1. **Impervious Surface** - Roads, sidewalks, parking lots
2. **Building** - Residential and commercial structures  
3. **Low Vegetation** - Grass, shrubs, small plants
4. **Tree** - Large woody vegetation
5. **Car** - Vehicles
6. **Background** - Other miscellaneous areas

### Dataset
- **Primary Dataset**: Vaihingen aerial imagery dataset with ground truth segmentation labels
- **Base Model**: CLIPSeg (CIDAS/clipseg-rd64-refined) - a vision-language model for zero-shot segmentation
- **Few-Shot Setup**: ~20 manually labeled images for fine-tuning, large independent test set for evaluation

## Project Structure

```
cv-for-urban-systems/
├── config.py                 # Central configuration file
├── code/                     # Core code modules
│   ├── crop_data.py              # Step 1: Data cropping
│   ├── prepare_ft_data.py        # Step 2: Data preparation
│   ├── data_augmentation.py      # Step 3: Data augmentation
│   ├── hyperparameter_search.py  # Step 4: Hyperparameter search
│   ├── final_train.py            # Step 5: Final training
│   ├── evaluate.py               # Step 6: Model evaluation
│   └── utils.py                  # Utility functions
├── data/                     # Data directories
│   └── Vaihingen/           
│       ├── top/             # Original aerial images
│       ├── ground_truth/    # Ground truth labels
│       └── finetune_data/   # Fine-tuning datasets
├── models/                   # Model storage
│   ├── clipseg_finetuned/   # Fine-tuned models
│   └── hyperparameter_search/ # Hyperparameter search results
└── output/                   # Output results
    └── evaluation/          # Evaluation results and visualizations
```

## Core Modules

### 1. Configuration Management (config.py)

**Purpose**: Centralized management of all project configuration parameters, including paths, hyperparameters, and dataset parameters.

**Key Configuration Sections**:
- **Path Configuration**: Project root, data directories, model directories, output directories
- **Dataset Configuration**: 
  - `finetune_pool_size`: 20 images (number of original images for fine-tuning)
  - `n_cv_folds`: 5 (cross-validation fold count)
  - `random_seed`: 42 (reproducibility)
  - `final_val_split`: 0.2 (validation split ratio)
- **Data Augmentation**: 
  - `num_augmentations_per_image`: 10 (augmentations per original image)
- **Hyperparameter Search Configuration**:
  - `n_trials`: 20 (number of hyperparameter combinations to try)
  - Search space: learning rate, Dice loss weight, batch size
- **Final Training Configuration**:
  - `num_epochs`: 100 (total training epochs)
  - `patience`: 10 (early stopping patience)
  - Checkpoint management settings
- **Evaluation Configuration**: Batch size, visualization parameters

### 2. Utility Functions (utils.py)

**Purpose**: Provides all common functionality needed across the project.

**Core Components**:
- **FineTuneDataset Class**: Custom dataset class handling image-mask-text triplets
- **Data Loading Functions**: `create_data_loader()`, collate functions
- **Model Management**: `create_model_and_optimizer()`, `save_model()`, `load_model()`
- **Loss Functions**: 
  - `dice_loss()`: Dice coefficient loss
  - `calculate_combined_loss()`: Combined BCE + Dice loss
- **Training Functions**: `train_one_epoch()`, `evaluate_model()`
- **Evaluation Metrics**: `calculate_iou()`, multiple segmentation metrics
- **Logging System**: Training process logging and checkpointing

### 3. Data Processing Pipeline

#### 3.1 Data Cropping (crop_data.py)
**Purpose**: Crops large original images into 512×512 patches for processing.

**Process**:
1. Read original aerial images and ground truth labels
2. Apply sliding window cropping (non-overlapping)
3. Save cropped image patches and corresponding labels

#### 3.2 Data Preparation (prepare_ft_data.py)
**Purpose**: Prepares fine-tuning datasets and generates binary masks for each class.

**Process**:
1. Randomly select specified number of images for fine-tuning pool
2. Reserve remaining images for test set
3. Generate individual binary mask files for each class
4. Create proper directory structure for training and testing

#### 3.3 Data Augmentation (data_augmentation.py)
**Purpose**: Expands limited training data to improve model generalization.

**Augmentation Strategies**:
- **Geometric Transformations**: Horizontal/vertical flips, rotations, translations, scaling, random crops
- **Photometric Transformations**: Brightness/contrast adjustments, color shifts, noise addition, blurring
- **Output Structure**:
  - K-fold cross-validation datasets (for hyperparameter search)
  - Full augmented dataset (for final training)

### 4. Model Training Pipeline

#### 4.1 Hyperparameter Search (hyperparameter_search.py)
**Purpose**: Automatically finds optimal hyperparameter combinations using Optuna framework.

**Search Strategy**:
- K-fold cross-validation for parameter evaluation
- Search space: learning rate (1e-6 to 5e-4), Dice weight (0.5 to 0.9), batch size (2, 4, 8)
- Early stopping to prevent overfitting
- Comprehensive logging of all trials

#### 4.2 Final Training (final_train.py)
**Purpose**: Trains the final model using optimal hyperparameters on full augmented dataset.

**Training Features**:
- Loads best hyperparameters from search results
- Trains on all augmented data with proper validation split
- Implements checkpointing and resume functionality
- Saves best model and detailed training logs
- Supports training process visualization

### 5. Model Evaluation (evaluate.py)

**Purpose**: Comprehensive evaluation comparing pre-trained and fine-tuned models.

**Evaluation Components**:
1. **Quantitative Evaluation**: 
   - IoU, Precision, Recall, F1-score, Pixel Accuracy for each class
   - Mean metrics across all classes
   - Statistical significance testing
2. **Qualitative Evaluation**:
   - 4-panel visualization comparisons (original, pre-trained output, fine-tuned output, ground truth)
   - Per-sample metric displays
3. **Output Results**:
   - CSV format evaluation summaries
   - JSON format complete results
   - PNG format visualization images
   - Comprehensive comparison plots

## Workflow

Complete project execution workflow:

```bash
# 1. Crop original data into 512x512 patches
python code/crop_data.py

# 2. Prepare fine-tuning datasets with binary masks
python code/prepare_ft_data.py

# 3. Generate augmented training data with K-fold splits
python code/data_augmentation.py

# 4. Search for optimal hyperparameters using cross-validation
python code/hyperparameter_search.py

# 5. Train final model with best hyperparameters
python code/final_train.py

# 6. Evaluate model performance with comprehensive metrics
python code/evaluate.py
```