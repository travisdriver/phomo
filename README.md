<div align="center">
<img src="assets/phomo.png" alt="logo" width="400">
<h1>Photoclinometry-from-Motion (PhoMo)</h1>

<a href="https://huggingface.co/datasets/travisdriver/phomo-data"><img src="https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg" alt="HuggingFace"></a>
<a href="https://arxiv.org/abs/2504.08252"><img src="https://img.shields.io/badge/arXiv-2504.08252-b31b1b" alt="arXiv"></a>

[Travis Driver](https://travisdriver.github.io/), [Andrew Vaughan](https://www.linkedin.com/in/andrewtvaughan/), [Yang Cheng](https://www-robotics.jpl.nasa.gov/who-we-are/people/yang_cheng/), [Adnan Ansar](https://www-robotics.jpl.nasa.gov/who-we-are/people/adnan_ansar/), [John Christian](https://ae.gatech.edu/directory/person/john-christian), [Panagiotis Tsiotras](https://ae.gatech.edu/directory/person/panagiotis-tsiotras)
</div>

#### This is the official repository for [Stereophotoclinometry Revisited](https://arxiv.org/abs/2504.08252), which has been accepted for publication to AIAA's [Journal of Guidance, Control, and Dynamics (JGCD)](https://arc.aiaa.org/loi/jgcd)

**Photoclinometry-from-Motion (PhoMo)** is a framework for _autonomous_ image-based surface reconstruction and characterization of small celestial bodies. PhoMo integrates photoclinometry into a structure-from-motion (SfM) pipeline that leverages deep learning-based keypoint extraction and matching (i.e., [RoMa](https://github.com/Parskatt/RoMa)) to enable _simultaneous_ optimization of the spacecraft pose, landmark positions, Sun vectors, and surface normals and albedos.

Data and results from the paper can be found on our [🤗Hugging Face page](https://huggingface.co/datasets/travisdriver/phomo-data).

If you find our datasets or results useful for your research, please use the following citation:

```bibtex
@article{driver2025phomo,
  title={Stereophotoclinometry Revisited},
  author={Driver, Travis and Vaughan, Andrew and Cheng, Yang, and Ansar, Adnan and Christian, John and Tsiotras, Panagiotis},
  journal={arXiv:2504.08252},
  year={2025},
  pages={1--45}
}
```

# Getting started

The following instructions will walk through how to install both GTSAM and PhoMo within the provided _conda_ environment, which does not require `sudo` access. The environment can be installed using the following commands: 

```bash
conda env create -f environment.yml
conda activate phomo
```

### 1. Install GTSAM from Source

GTSAM and its [Python wrapper](https://github.com/borglab/gtsam/blob/develop/python/README.md) can be installed by cloning the GTSAM repository to your desired location and running the following from __GTSAM's root directory__ (making sure the `phomo` environment is activated):

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGTSAM_BUILD_PYTHON=ON
cmake --build build -j8 --target install python-install
```

### 2. Install PhoMo

Run the following in the root of this repository to install PhoMo:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
cmake --build build -j8 --target install python-install
```

Unit tests can be run from the `build/` directory by running `make check`. 

### 3. Run example

Example data for a half-resolution version of the Cornelia experiment can be downloaded by running `python scripts/download_example_data.py`. PhoMo can be used to refine the initial map by running

```bash
python run.py --init_model_path data/reconstructions/cornelia/lunar_lambert_if_corrected_half_res/init --config vesta_lunar_lambert --output_path results/cornelia_lunar_lambert_half_res
```

The PhoMo map can then be rendered and compared against the actual images using

```bash
python render.py --model_path results/cornelia_lunar_lambert_half_res --images_path data/images/cornelia --config vesta_lunar_lambert
```

Full resolution data for Cornelia, as well as data for the Ahuna Mons and Ikapati experiments, can be found on PhoMo's  🤗[Huggingface page](https://huggingface.co/datasets/travisdriver/phomo-data).
