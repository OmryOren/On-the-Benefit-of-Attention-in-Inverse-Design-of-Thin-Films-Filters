# On-the-Benefit-of-Attention-in-Inverse-Design-of-Thin-Films-Filters

## Abstract
Attention layers are a crucial component in many modern deep learning models, particularly those used in natural language processing and computer vision. 
Attention layers have been shown to improve the accuracy and effectiveness of various tasks, such as machine translation, image captioning, etc.
Here, the benefit of attention layers in designing optical filters based on a stack of thin film materials is investigated.
The superiority of Attention layers over fully-connected Deep Neural Networks is demonstrated for this task.

## Usage Instructions
1. **Generate Dataset:**
  - Open the file named `5_layers_data_generator.py` (for dataset of 5 layers in each sample) or the file named `unknown_layers_data_generator.py` (for dataset of unknown amount of layers between 3-5 in each sample).
  - Copy the file `maretials.csv` to the same directiry as the code file.
  - Run this code to generate the dataset of random filters, and set the amount of samples by setting the input `num` of the function `examples(materials_dict, num)`.
