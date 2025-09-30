<div align="center">
<img src="assets/phomo.png" alt="VGGT Logo" width="400">
<h1>Photoclinometry-from-Motion (PhoMo)</h1>

<a href="https://huggingface.co/datasets/travisdriver/phomo-data"><img src="https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg" alt="HuggingFace"></a>
<a href="https://arxiv.org/abs/2504.08252"><img src="https://img.shields.io/badge/arXiv-2504.08252-b31b1b" alt="arXiv"></a>

[Travis Driver](https://travisdriver.github.io/), [Andrew Vaughan](https://www.linkedin.com/in/andrewtvaughan/), [Yang Cheng](https://www-robotics.jpl.nasa.gov/who-we-are/people/yang_cheng/), [Adnan Ansar](https://www-robotics.jpl.nasa.gov/who-we-are/people/adnan_ansar/), [John Christian](https://ae.gatech.edu/directory/person/john-christian), [Panagiotis Tsiotras](https://ae.gatech.edu/directory/person/panagiotis-tsiotras)
</div>

#### This is the official repository for [Stereophotoclinometry Revisited](https://arxiv.org/abs/2504.08252), which is currently under review for publication to AIAA's [Journal of Guidance, Control, and Dynamics (JGCD)](https://arc.aiaa.org/loi/jgcd)

**Photoclinometry-from-Motion (PhoMo)** is a framework for _autonomous_ image-based surface reconstruction and characterization of small celestial bodies. PhoMo integrates photoclinometry into a structure-from-motion (SfM) pipeline that leverages deep learning-based keypoint extraction and matching (i.e., [RoMa](https://github.com/Parskatt/RoMa)) to enable _simultaneous_ optimization of the spacecraft pose🛰️, landmark positions, Sun vectors, and surface normals and albedos.

We are currently working to release the code as soon as possible. Data and results from the paper can be found on our [🤗Hugging Face page](https://huggingface.co/datasets/travisdriver/phomo-data).

If you find our datasets or code useful for your research, please use the following citation:

```bibtex
@article{driver2025phomo,
  title={Stereophotoclinometry Revisited},
  author={Driver, Travis and Vaughan, Andrew and Cheng, Yang, and Ansar, Adnan and Christian, John and Tsiotras, Panagiotis},
  journal={arXiv:2504.08252},
  year={2025},
  pages={1--45}
}
```
