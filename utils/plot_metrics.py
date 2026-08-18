import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to prevent freezing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def generate_and_save_plots(y_true, y_pred, y_prob, num_classes, phase_name, dataset, embed, algo):
    """
    Generate Confusion Matrix and ROC Curve for a specific model evaluation.
    Properly isolates plots into outputs/plots/phase/dataset/embedding/
    """
    try:
        # Create directory structure
        if isinstance(embed, tuple):
            embed_str = "_".join(embed)
        else:
            embed_str = str(embed)
            
        base_dir = os.path.join("outputs", "plots", phase_name, dataset, embed_str)
        os.makedirs(base_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {algo} + {embed_str} ({dataset})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{algo}_confusion_matrix.png"))
        plt.close('all')
        
        # 2. ROC Curve
        # Binary Classification Only for simple visualization
        if num_classes == 2:
            if y_prob is not None:
                if y_prob.ndim > 1 and y_prob.shape[1] == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob
                
                fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve: {algo} + {embed_str} ({dataset})')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(base_dir, f"{algo}_roc_curve.png"))
                plt.close('all')
    except Exception as e:
        print(f"Plotting Error for {algo}: {e}")
