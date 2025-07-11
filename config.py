from pathlib import Path

# ==============================================================================
# 1. Core Path Configuration
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

ORIGINAL_DIR = DATA_DIR / "Vaihingen"
# CROPPED_IMAGES_DIR = ORIGINAL_DIR / "top_cropped_512"
# CROPPED_LABELS_DIR = ORIGINAL_DIR / "ground_truth_cropped_512"


FINETUNE_DATA_DIR = ORIGINAL_DIR / "finetune_data"
# FINETUNE_POOL_IMG_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "images"
# FINETUNE_POOL_MASK_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "masks"

FINETUNED_MODEL_DIR = MODELS_DIR / "clipseg_finetuned"
HYPERPARAMETER_SEARCH_DIR = MODELS_DIR / "hyperparameter_search"

# ==============================================================================
# 2. General Data & Model Configuration
# ==============================================================================
# CLIPSeg pretrained model
PRETRAINED_MODEL = "CIDAS/clipseg-rd64-refined"

# Urban segmentation classes
URBAN_CLASSES = [
    'impervious surface', 'building', 'low vegetation', 'tree', 'car', 'background'
]


CROP_SIZE = 512  # Image crop size

# Class color definitions (RGB format) - used for ground truth mask extraction and visualization
CLASS_COLORS_RGB = {
    'impervious surface': (255, 255, 255),  # White
    'building': (0, 0, 255),                # Blue
    'low vegetation': (0, 255, 255),        # Cyan
    'tree': (0, 255, 0),                    # Green
    'car': (255, 255, 0),                   # Yellow
    'background': (255, 0, 0)               # Red
}

# Dataset parameters
DATASET_CONFIG = {
    'finetune_pool_size': 20,  # LAPTOP: 4 | SERVER: 20 (number of original images)
    'n_cv_folds': 5,           # LAPTOP: 2 | SERVER: 5 (cross-validation fold count)
    'random_seed': 42,         # Random seed
    'final_val_split': 0.2     # Validation split ratio for final training (original image level)
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'num_augmentations_per_image': 10,  # LAPTOP: 3 | SERVER: 10
}

# ==============================================================================
# 3. Hyperparameter Search Configuration
# ==============================================================================
# For hyperparameter_search.py
SEARCH_CONFIG = {
    # Optuna framework settings
    'n_trials': 20,               # LAPTOP: 2 | SERVER: 20 (total number of hyperparameter combinations to try)
    'study_name': 'clipseg_cv_search_laptop',
    'direction': 'minimize',     # Optimization direction: minimize validation loss

    # Training settings for each trial
    'num_epochs': 20,             # LAPTOP: 2 | SERVER: 10 (training epochs per trial)
    'patience': 5,            # LAPTOP: 3 | SERVER: 5 (early stopping patience)

    # Hyperparameter search space
    'space': {
        'learning_rate': {'type': 'loguniform', 'low': 1e-6, 'high': 5e-4},
        'dice_weight':   {'type': 'uniform', 'low': 0.5, 'high': 0.9},
        'batch_size':    {'type': 'categorical', 'choices': [2, 4, 8]},
    }
}

# ==============================================================================
# 4. Final Model Training Configuration
# ==============================================================================
# For final_train.py
FINAL_TRAIN_CONFIG = {
    'num_epochs': 100,             # LAPTOP: 5 | SERVER: 50 (total training epochs for final model)
    'patience': 10,                # Early stopping patience (epochs without validation loss improvement)
    'min_delta': 1e-4,            # Minimum improvement threshold
    
    # Note: validation split is already defined in DATASET_CONFIG['final_val_split']
    # Data is pre-split in data_augmentation.py
    
    # Note: the following learning_rate, batch_size, dice_weight
    # will be overwritten by the best values found by hyperparameter search.
    # These values are only used for debugging or as fallback when no search results exist.
    'default_learning_rate': 5e-5,
    'default_batch_size': 4,
    'default_dice_weight': 0.8,
    
    # Checkpoint configuration
    'checkpoint_interval': 5,      # Save checkpoint every N epochs
    'max_checkpoints': 3,          # Maximum number of checkpoints to keep
    'resume_from_checkpoint': True,  # Whether to automatically resume from checkpoint
}

# ==============================================================================
# 5. System & Evaluation Configuration
# ==============================================================================
RANDOM_SEED = DATASET_CONFIG['random_seed']
DEVICE = 'cuda'  # Global device configuration: 'auto', 'cuda', 'cpu', 'mps'
NUM_WORKERS = 0 # Set to 0 for debugging convenience

EVALUATION_CONFIG = {
    'default_model_path': FINETUNED_MODEL_DIR / "best_model",
    'default_test_data': FINETUNE_DATA_DIR / "test" / "images",
    'default_output_dir': OUTPUT_DIR / "evaluation",
    'batch_size': 2,
    'num_visualization_samples': 20,     # Number of samples to visualize in evaluation
    'figure_size': (16, 16),
    'visualization_dpi': 150,
}

if __name__ == "__main__":
    print("✅ config.py: Configuration file loaded successfully.")
    print(f"  - Search trials: {SEARCH_CONFIG['n_trials']}")
    print(f"  - Final training epochs: {FINAL_TRAIN_CONFIG['num_epochs']}")
