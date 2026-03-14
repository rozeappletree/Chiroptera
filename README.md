# Chiroptera (Notebook Quick Guide)

Only the two notebooks below are needed.

## 1) `Models_V1/notebooks/Master_Training_n_Testing _notebook.ipynb`

### Change (first config cell)
- Set dataset paths (spectrogram folder, features CSV, test audio folder).
- Set training options (model type, batch size, epochs, learning rate).
- Set output model path (where checkpoint is saved).

### Use
- Open notebook In Kaggle.
- Edit the first config cell.
- Run all cells top-to-bottom.

## 2) `Models_V1/notebooks/Spectrogram_Display.ipynb`

### Change (first path/config cell)
- Set `DATASET_ROOT` to your audio root folder.
- Set annotation file path (`.json`) from your dataset.
- (Optional) set display/sample limits.

### Use
- Open notebook in Kaggle.
- Edit path variables in the first config cell.
- Run all cells to visualize spectrograms + boxes.

