# AI for Digital Pathology

A hands-on YouTube series teaching AI-based computational pathology from first principles. We go from raw tissue images to survival prediction using foundation models, attention mechanisms, and spatial analysis - with real clinical datasets throughout.

**YouTube Series:** [AI in Digital Pathology - DigitalSreeni](https://www.youtube.com/@DigitalSreeni)

---

## Series Overview

| Video | Title | Key Concepts | Dataset |
|---|---|---|---|
| 1 | Why Now? | Foundation models, embeddings, series roadmap | None |
| 2 | Patch Embeddings | Phikon-v2, CLS token, UMAP, cosine similarity | PatchCamelyon |
| 3 | Tissue Classification | Weak supervision, mean pooling, logistic regression | BR2082c TMA |
| 4 | ABMIL | Attention pooling, MIL, spatial heatmaps | BR2082c TMA |
| 5a | WSI Mechanics | OpenSlide, pyramid structure, Otsu filtering, tiling | CMU-1.svs |
| 5b | Spatial TME Analysis | Tumor microenvironment, TSR, Kaplan-Meier | SurGen SR386 |
| 6 | Survival Prediction | Cox model, risk stratification, KM curves | SurGen SR386 |

---

## Notebooks

```
notebooks/
├── video2_patch_embeddings.ipynb
├── video3_tma_classification.ipynb
├── video4_abmil.ipynb
├── video5a_wsi_processing.ipynb
├── video5b_spatial_analysis.ipynb
└── video6_survival_prediction.ipynb
```

---

## Datasets

### PatchCamelyon (Video 2)
Breast lymph node H&E patches, 96x96 px, binary tumor/normal labels.
Download: https://github.com/basveeling/pcam

```
data/pcam/camelyonpatch_level_2_split_test_x.h5
data/pcam/camelyonpatch_level_2_split_test_y.h5
```

### BR2082c TMA (Videos 3 and 4)
187 breast tissue cores, 3000x3000 px, H&E, spanning carcinoma / DCIS / normal/benign.
Download: https://www.tissuearray.com/tissue-arrays/Breast/BR2082c

```
images/P1522Q0001-0008/BR2082c specs.xlsx
images/P1522Q0001-0008/core_images/*.png
```

### CMU-1.svs (Video 5a)
Whole slide image used for technical demonstration of WSI processing mechanics. Tissue type uncharacterized. Distributed by the OpenSlide project as test data.
Download: https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs

```
images/CMU-1.svs
```

### SurGen SR386 (Videos 5b and 6)
427 primary colorectal cancer WSIs with 5-year survival data, KRAS/NRAS/BRAF mutation status, MMR status, and staging. We use pre-extracted UNI patch embeddings rather than downloading the raw WSI files (100+ GB).

Pre-extracted embeddings (16.8 GB): https://doi.org/10.5281/zenodo.14047723
Labels and dataset info: https://github.com/CraigMyles/SurGen-Dataset
Paper: Myles C et al. GigaScience 2025. https://doi.org/10.1093/gigascience/giaf086

```
data/surgen/SurGen_UNI_patch_embeddings.zip
data/surgen/SR386_labels.csv
```

---

## Model

All embedding extraction uses **Phikon-v2** (owkin/phikon-v2 on HuggingFace).

- Architecture: ViT-Large
- Training: DINOv2 self-supervised on 450 million pathology patches
- Output: 1024-dimensional CLS token embedding per 256x256 patch
- Access: Fully open, no institutional email required

```python
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model     = AutoModel.from_pretrained("owkin/phikon-v2")
```

Videos 5b and 6 use pre-extracted UNI embeddings from the SurGen Zenodo record to avoid downloading 100+ GB of raw WSI files. UNI is also a 1024-dim ViT-Large pathology foundation model.

---

## Environment

```bash
conda create -n torch-gpu-pathology python=3.11
conda activate torch-gpu-pathology
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers openslide-bin zarr lifelines umap-learn
pip install scikit-learn pandas matplotlib pillow tqdm h5py
```

Tested on: Windows 11, NVIDIA RTX 4000 Ada (20GB VRAM), CUDA 12.1

---

## Project Structure

```
AI_in_Digital_Pathology/
├── images/
│   ├── CMU-1.svs
│   └── P1522Q0001-0008/
│       ├── BR2082c specs.xlsx
│       └── core_images/
├── data/
│   ├── pcam/
│   └── surgen/
├── notebooks/
└── results/
    ├── video2/
    ├── video3/
    ├── video4/
    ├── video5a/
    ├── video5/
    └── video6/
```

---

## License

Code in this repository is released under the MIT License.
Dataset licenses apply separately - see each dataset source for terms.

---

## Author

**Sreenivas Bhattiprolu**
YouTube: [DigitalSreeni](https://www.youtube.com/@DigitalSreeni)
GitHub: [github.com/bnsreenu](https://github.com/bnsreenu)
