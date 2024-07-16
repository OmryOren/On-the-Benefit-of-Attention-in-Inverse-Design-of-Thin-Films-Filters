# On-the-Benefit-of-Attention-in-Inverse-Design-of-Thin-Films-Filters

## Abstract
Attention layers are a crucial component in many modern deep learning models, particularly those used in natural language processing and computer vision. 
Attention layers have been shown to improve the accuracy and effectiveness of various tasks, such as machine translation, image captioning, etc.
Here, the benefit of attention layers in designing optical filters based on a stack of thin film materials is investigated.
The superiority of Attention layers over fully-connected Deep Neural Networks is demonstrated for this task.

## Instructions

1. **Generate Dataset:**
  - Open `5_layers_data_generator.py` (for datasets with 5 layers per sample) or `unknown_layers_data_generator.py` (for datasets with an unknown number of layers between 3-5 per sample).
  - Copy `materials.csv` to the same directory as the code file.
  - Run the code to generate the dataset of random filters. Set the number of samples by specifying the input `num` in the `examples(materials_dict, num)` function.

2. **Transformer Model:**
  - Open `5_layers_transformer_model.ipynb` (for datasets with 5 layers per sample) or `unknown_layers_transformer_model.ipynb` (for datasets with an unknown number of layers between 3-5 per sample).
  - Update the `files_path` variable at the top of the file to point to the dataset directory, and change the `new_data_files_path` variable to the desired results directory.
  - Run all the cells in the notebook sequentially.
