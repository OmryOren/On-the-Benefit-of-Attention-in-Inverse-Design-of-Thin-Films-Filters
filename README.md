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

2. **DNN & Transformer Model:**
  - DNN: Open `5_layers_DNN_model.ipynb` (for datasets with 5 layers per sample) or `unknown_layers_DNN_model.ipynb` (for datasets with an unknown number of layers between 3-5 per sample).
  - Transformer :Open `5_layers_transformer_model.ipynb` (for datasets with 5 layers per sample) or `unknown_layers_transformer_model.ipynb` (for datasets with an unknown number of layers between 3-5 per sample).
  - Update the `files_path` variable at the top of the file to point to the dataset directory, and change the `new_data_files_path` variable to the desired results directory.
  - Run all the cells in the notebook sequentially.

3. **Figure Codes:**
  - The files `Figure_Code-Attention_Scores.py`, `Figure_Code-Multiple_Runnings_Histogram.py`, and `Figure_Code-Waves_Graphs.py` are used to process the data from the result files of the model and generate the figures for the paper.
  - To run these codes, save all result files from the DNN models and Transformer models codes in a directory and update the `data_folder` variable to point to that directory.
  - Update the `output_folder` variable to point to the directory where the figures will be saved.
  - For `Figure_Code-Multiple_Runnings_Histogram.py`, each model must be run 10 times with different run numbers from 1 to 10. The `running_number` variable is located at the top of each notebook.
