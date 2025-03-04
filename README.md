# World Model Imagination as Explanations

An extended and slightly modified implementation of [DreamerV3][paper] with the goal to use the imaginations as an explanation.

In case of LFS bandwith limitations there is a mirror at:
https://gitlab.kit.edu/nils.wenninghoff/dreamer_explanation

# Instructions

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python training.py 
```

Evaluation script:
```sh
python evaluation.py 
```

# About This Fork
This project is a fork of [Original Repository][repo] by Danijar Hafner.
It includes modifications by Nils Wenninghoff to implement additional reporting functionality and introduce new explanation feature.
The original project is licensed under MIT, and this fork continues under the same license.



[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104v1.pdf
[repo]: https://github.com/danijar/dreamerv3
