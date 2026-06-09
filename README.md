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
| 5b | Spatial TME Analysis | Tumor microenvironment, TSR, spatial features, survival correlation | SurGen SR386 |
| 6 | Survival Prediction | Mean pooling, PCA, logistic regression, risk stratification | SurGen SR386 |

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

- **Download:** https://github.com/basveeling/pcam
- **Direct links:**
  - Test images: https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz
  - Test labels: https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz

```
data/pcam/camelyonpatch_level_2_split_test_x.h5
data/pcam/camelyonpatch_level_2_split_test_y.h5
```

### BR2082c TMA (Videos 3 and 4)
187 breast tissue microarray cores, 3000x3000 px, H&E, spanning carcinoma, DCIS, and normal/benign tissue.

- **Product page:** https://www.tissuearray.com/tissue-arrays/Breast/BR2082c
- **Note:** Registration required. Request a quote for the digital image set. Core annotations are provided in the accompanying specs spreadsheet.

```
images/P1522Q0001-0008/BR2082c specs.xlsx
images/P1522Q0001-0008/core_images/*.png
```

### CMU-1.svs (Video 5a)
Whole slide image used for technical demonstration of WSI processing mechanics. Tissue type uncharacterized. Distributed freely by the OpenSlide project as a scanner compatibility test file.

- **Direct download:** https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
- **Size:** 169 MB
- **Format:** Aperio SVS, 46,000 x 32,914 px at level 0, 0.499 um/px (~20x equivalent), 3 pyramid levels
- **Citation:** Goode A et al. "OpenSlide: A vendor-neutral software foundation for digital pathology." Journal of Pathology Informatics 2013;4:27. https://doi.org/10.4103/2153-3539.119005

```
images/CMU-1.svs
```

### SurGen SR386 (Videos 5b and 6)
427 primary colorectal cancer whole slide images with 5-year survival data, KRAS/NRAS/BRAF mutation status, MMR status, and clinical staging. We use pre-extracted UNI patch embeddings (16.8 GB) rather than the raw WSI files (100+ GB).

- **Pre-extracted UNI embeddings (16.8 GB):** https://doi.org/10.5281/zenodo.14047723
  - File: `SurGen_UNI_patch_embeddings.zip`
  - Contains embeddings for both SR386 (427 slides) and SR1482 (593 slides). We use SR386 only.
- **Labels CSV and dataset info:** https://github.com/CraigMyles/SurGen-Dataset
  - File: `SR386_labels.csv` (available from the Zenodo record or the GitHub repository)
- **Dataset paper:** Myles C et al. "SurGen: 1020 H&E-stained whole-slide images with survival and genetic markers." GigaScience 2025. https://doi.org/10.1093/gigascience/giaf086
- **Embeddings paper:** Myles C et al. "Leveraging foundation models for enhanced detection of colorectal cancer biomarkers in small datasets." MIUA 2024, Springer LNCS 14859, pp.329-343. https://doi.org/10.1007/978-3-031-66958-3_22
- **Survival breakdown:** 264 survived 5 years (62%), 161 died (38%). Stage 1-4 distribution: 60, 153, 120, 20. MMR loss: 32 cases (7.5%).

```
data/surgen/SurGen_UNI_patch_embeddings.zip
data/surgen/SR386_labels.csv
```

---

## Models

### Phikon-v2 (Videos 2, 5a)
Used for all embedding extraction in notebooks 1 through 5a.

- **HuggingFace:** https://huggingface.co/owkin/phikon-v2
- **Architecture:** ViT-Large, DINOv2 self-supervised training on 450 million pathology patches
- **Output:** 1024-dimensional CLS token embedding per 256x256 patch
- **Access:** Fully open, no institutional email or agreement required

```python
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model     = AutoModel.from_pretrained("owkin/phikon-v2")
```

### UNI (Videos 5b, 6)
Used in the pre-extracted SurGen embeddings. We do not run UNI directly - we load the pre-extracted embeddings from the Zenodo record.

- **HuggingFace:** https://huggingface.co/MahmoodLab/uni
- **Architecture:** ViT-Large, trained on 100,000+ pathology slides
- **Output:** 1024-dimensional embedding per 256x256 patch at 40x magnification
- **Access:** Gated - institutional email required at time of writing. Use the pre-extracted Zenodo embeddings to avoid this.

---

## Environment

```bash
conda create -n torch-gpu-pathology python=3.11
conda activate torch-gpu-pathology

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install transformers openslide-bin zarr
pip install lifelines umap-learn scikit-learn
pip install pandas matplotlib pillow tqdm h5py scipy
```

**Tested on:** Windows 11, NVIDIA RTX 4000 Ada (20 GB VRAM), CUDA 12.1

**SSL workaround (corporate proxy):** If you are behind a Zscaler or similar proxy that intercepts SSL, add the following at the top of each notebook:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

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

## Citations

If you use code or methods from this series in your own work, please cite the relevant dataset and model papers:

**OpenSlide**
Goode A, Gilbert B, Harkes J, Jin D, Satyanarayanan M. "OpenSlide: A vendor-neutral software foundation for digital pathology." Journal of Pathology Informatics 2013;4:27. https://doi.org/10.4103/2153-3539.119005

**SurGen dataset**
Myles C, Um IH, Harrison DJ, Harris-Birtill D. "SurGen: 1020 H&E-stained whole-slide images with survival and genetic markers." GigaScience 2025. https://doi.org/10.1093/gigascience/giaf086

**SurGen UNI embeddings**
Myles C et al. "Leveraging foundation models for enhanced detection of colorectal cancer biomarkers in small datasets." Medical Image Understanding and Analysis (MIUA) 2024, Springer LNCS 14859, pp.329-343. https://doi.org/10.1007/978-3-031-66958-3_22

**PatchCamelyon**
Veeling BS, Linmans J, Winkens J, Cohen T, Welling M. "Rotation Equivariant CNNs for Digital Pathology." MICCAI 2018. https://doi.org/10.1007/978-3-030-00934-2_24

---

## License

Code in this repository is released under the MIT License.
Dataset licenses apply separately - see each dataset source for terms.

---

## Author

**Sreenivas Bhattiprolu**
YouTube: [DigitalSreeni](https://www.youtube.com/@DigitalSreeni)
GitHub: [github.com/bnsreenu](https://github.com/bnsreenu)
