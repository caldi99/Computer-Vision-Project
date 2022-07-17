# Instructions

- Use `format_dataset.m` to resize the images and labels and save them into the correct format

- Use `clean_dataset.m` to clean the dataset from black masks that can occur in case of an error in the dataset

- Finally you can train the model by run `train_DA.m`, this script will also create the augumented training set, save the model and save the masks

- Use `convert_model.m` to convert your model form `.mat` to `.onnx`

## Augmentation Methods

In the functions `imagesTrasformation_new.m`, `contrast_blur_h_new.m` and `shadows_new.m` are implemented data augmentation methods