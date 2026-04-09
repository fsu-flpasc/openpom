# Olfactorial Perceptronics

Predicts odor qualities from molecular SMILES strings using an ensemble of pretrained MPNN models based on the [Principal Odor Map](https://doi.org/10.1126/science.ade4401) paper by Lee et al. (2023).

## Requirements

- Python 3.10
- Git LFS

## Setup

1. Clone this repository
```bash
   git clone git@github.com:fsu-flpasc/openpom.git
   cd openpom
```

2. Create and activate a virtual environment
```bash
   ~/.pyenv/versions/3.10.14/bin/python -m venv .venv
   source .venv/bin/activate
```

3. Install dependencies
```bash
   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
   pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/torch-2.2/cpu/repo.html
   pip install -r requirements.txt
```

4. Clone the pretrained model weights (requires Git LFS)
```bash
   git clone https://github.com/BioMachineLearning/openpom.git openpom-repo
   cd openpom-repo
   git lfs pull
   cd ..
```

## Usage

```bash
python predict.py
```

You will be prompted to enter a SMILES string. The script will look up the molecule name via PubChem and return the top 10 predicted odor qualities using an ensemble of 10 pretrained models.

## Example

Enter SMILES: COc1cc(C=O)ccc1O
Looking up molecule name...
Top predicted odors for vanillin (COc1cc(C=O)ccc1O):
vanilla: 81.8%
sweet: 73.0%
spicy: 55.2%

## Credits

Pretrained models from [BioMachineLearning/openpom](https://github.com/BioMachineLearning/openpom), based on the original [ARY2260/openpom](https://github.com/ARY2260/openpom) implementation.
