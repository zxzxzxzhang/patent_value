## License

This code is provided under a **pre-release license**. 

- **Usage**: The code may only be used for academic or non-commercial purposes.
- **Redistribution**: Redistribution or modification is not permitted until the associated research paper is published.

Please refer to the [LICENSE](./LICENSE) file for full details.

For inquiries or specific permissions, contact us at zhangx2293@gmail.com with the subject "Pre-release Code Inquiry."




# --------------------------------------------------------

# Predict patent value

## Project Structure

```
\patent value
|-- DataAugmentor.py         # Data augmentation tool
|-- DataLoader.py            # Data loading 
|-- Fine_tune_distilbert.py  # Fine-tuning DistilBERT script
|-- run.py                   # Main entry point of the project
|-- translate
|   |-- df_translator.py     # DataFrame translation module
|   |-- translate.py         # translation
|-- models
    |-- model_gat.py         # Graph Attention Network (GAT) implementation
    |-- model_Gated.py       # Gated Graph Neural Network implementation
    |-- model_gcn.py         # Graph Convolutional Network (GCN) implementation
    |-- model_multi_sage.py  # Multi-layer GraphSAGE implementation
    |-- model_sage.py        # Basic GraphSAGE model implementation
    |-- __init__.py          # Module initializer
```

---

## Usage
Running the Main Script
Use run.py as the entry point:
```bash
python run.py
```
Main model:
```bash
python ./models/model_multi_sage.py
```
