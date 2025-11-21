Paper: "MDL-HTI: A Multimodal Deep Learning Approach for Predicting Herb-Target Interactions" (DOI: 10.1007/s12539-025-00772-w)Published in Interdiscip Sci Comput Life Sci: https://link.springer.com/article/10.1007/s12539-025-00772-w

# MDL-HTI
This repository contains the data and Python scripts related to the manuscript titled "MDL-HTI: Multimodal deep learning approach for predicting herb-target interactions". It provides the necessary data sources for model training and evaluation, along with Python scripts that are used for data preprocessing, model implementation and training procedures.
## Requirements
* Python 3.7+
* numpy 1.21.0
* torch 1.10.0
* cuda 1.10+
## Dataset Description

### Current Availability  
We currently provide two datasets for standard training and testing:  
- **HTI-CPL Dataset**  
- **HTI-CPL_comparison Dataset**  

**Download Instructions:**  
Both datasets can be downloaded from [this link](https://pan.baidu.com/s/1tldU38kV7NJ16oJHTXp1iA?pwd=g44d). After downloading:  
1. Place the `train/valid/test` files in the `data/ddb` directory.  
2. Place the `codes_one_hot` file in the `data/ddb/herb_data` directory.  

### Anonymization Protocol  
To ensure research integrity during the pre-publication period:  
- Traditional Chinese Medicines (TCMs) and targets are anonymized using **numerical identifiers** (e.g., 1, 2, 3).  
- TCM ingredient data and target pathway/ligand information are encoded as **one-hot vectors**.  

### Post-Publication Updates  
Upon paper acceptance:  
1. Full mappings between numerical identifiers and real entity names will be released.  
2. Complete relationship matrices for:  
   - TCM ingredients  
   - Target pathways  
   - Ligand interactions  

*This anonymization strategy aligns with journal preprint policies. We appreciate your understanding.*  
## Execution
To run MDL-HTI, execute the sample command:
```python
python main.py --dataset <dataset> --clustering <0/1> --hidden_dim <hidden dimension> --lr <learning rate> --weight_decay <weight decay> --num_heads <number of heads> --num_layers <number of layers> --dropout <dropout> --context_hops <context_hops> --max_path_len <max_path_len> --path_samples <path_samples> --cluster_coeff <clustering coefficients> --num_clusters <number of clusters> --gpu_num <gpu_id>

Example: python main.py --dataset ddb --clustering 1 --hidden_dim 128 --lr 0.0001 --weight_decay 0.0007 --num_heads 8 --num_layers 4 --dropout 0.1 --context_hops 4 --max_path_len 5 --path_samples 5 --cluster_coeff 0.5 --num_clusters 25 --gpu 0
```
To conduct experiments with different datasets:  
1. Use `graph_classifier_1.py` for the **HTI-CPL Dataset**  
2. Use `graph_classifier_2.py` for the **HTI-CPL_comparison Dataset**  

The relevant dataset-specific parameters are pre-configured in the respective classifier files. Additionally, adjustments for sampling strategies can be made in [`src/sampler_graphs.py`], where critical parameters are already annotated for customization.
