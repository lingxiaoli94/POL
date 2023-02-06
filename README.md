# Learning Proximal Operators to Discover Multiple Optima

This repository contains the Source code for the ICLR 2023 paper ([link](https://arxiv.org/abs/2201.11945)) "Learning Proximal Operators to Discover Multiple Optima" by Lingxiao Li, Noam Aigerman, Vladimir G. Kim, Jiajin Li, Kristjan Greenewald, Mikhail Yurochkin, Justin Solomon.

## Dependencies

The only core dependency is PyTorch. We will also need the CUDAToolkit to compile the CUDA library `geomlib` for symmetry detection only (CUDAToolkit should come with pytorch if installed using conda).

Additional dependencies:
- Logging + plotting:
    * h5py
    * tqdm
    * matplotlib
    * jupyterlab + ipywidgets
    * tensorboard
- Symmetry detection dependencies:
    * pybind11
    * ninja
    * meshio
    * pyvista (for visualization)
- Object detection dependencies:
    * PIL 
    * fiftyone 
    * albumentations 

<!--
To install, run
```
conda install pybind11 ninja tqdm h5py matplotlib jupyterlab tensorboard
pip install ipywidgets Pillow fiftyone timm albumentations meshio pyvista
```
-->

## Code organization
The core implementation is contained in `pol` package (`pol` is short for proximal operator learning). The organization of `pol` package is as follows.
- `problems` folder contains classes inherited from `ProblemBase` (defined in `problem_base.py`) specialized for different applications. We implemented the following applications.
    * `AnalyticalProblem`: the problem class where the objective function is given analytically and its evaluation does not have stochasticity. This class is used in the "sampling from conic sections" experiment (Section 5.1) as well as the experiments in Section D.2, D.3 of the paper.
    * `SupervisedLearningProblem`: the problem class for supervised learning, where the objective function can only be evaluated stochastically on batches of the dataset. This class is used in the "sparse recovery" experiment (Section 5.2) in the paper.
    * `MaxCutProblem`: the problem class for the "rank-2 relaxation of max-cut" experiment (Section 5.3) of the paper.
    * `SymmetryDetection`: the problem class for the "symmetry detection of 3D shapes" experiment (Section 5.4) of the paper.
    * `ObjectDetection`: the problem class for the "object detection in images" experiment (Section 5.5) of the paper.
- `solvers` folder contains a list of solver classes. *Universal* solvers (child classes of `UniversalSolverBase`) are those that can generalize to new problems with different parameters.
    * `ParticleDescent`: a baseline solver that simply runs gradient descent independently on initial particles.
    * `POL`: the proximal operator learning solver, a universal solver that is the main contribution of the paper.
    * `GOL`: the gradient operator learning solver, a universal solver proposed in the paper as an alternative to compare against.
    * `FRCNN` and `FixedNumberSolver`: specialized universal solvers for object detection only for comparison purposes.
    * `configs` folder contains classes used to configure experiments.
    * `nn` folder contains neural network architectures used.
- `datasets` folder contains classes used to prepare the datasets.
- `runners` folder contains runner classes used to run experiments (e.g. saving/loading checkpoints, training for loops).

In addition to `pol` package, there are a few other folders in the root directory:
- `assets` folder is used to store assets (e.g. MCB dataset for symmetry detection).
- `notebooks` folder contains jupyter notebooks used to make plots in the paper.
- `geomlib` is a standalone package used in symmetry detection to query the distance field of points to a 3D mesh.
- `tests` folder contains the working directories of all experiments (running experiments is detailed in the next section).


## How to run
First, you will need to install the package `pol` in order to run it. This can be done by either `pip install -e .` (using pip) or `conda develop .` (using conda).
Either way, you will be able to modify the source code and the changes will be reflected immediately the next time you use the package `pol`.

The `tests` folder contains the working directories of the experiments. The folders `analytical`, `linear_regression`, `maxcut`, `symmetry_detection`, and `objdetect` correspond to the five experiments in Section 5 of the paper respectively. In each working directory, there is a `config.py` file that is the entry point of the corresponding experiment. There is also a `script.sh` which includes the command line used to run the experiment for each method. Evaluation scripts named `eval.py` are also included (though not so well-maintained).

For example, to run the "sampling from conic sections" experiment, from the project root directory, execute
```
cd tests/analytical
python config.py --problem_list=conic --method_list=pol_res_lot --train_step=200000 --restart
```

If you wish to run the symmetry detection experiment (Section 5.4 in the paper), the setup is slightly more involved as we need to compile the CUDA library `geomlib`.
Be sure to install `pybind11` and `ninja` through conda, in addtional to PyTorch (we need the CUDAToolkit that comes with it).
Then run the script `install.sh` or adapt the script to your needs.
You also need to download the [MCB dataset A](https://engineering.purdue.edu/cdesign/wp/a-large-scale-annotated-mechanical-components-benchmark-for-classification-and-retrieval-tasks-with-deep-neural-networks/) and then extract the `.tar.gz` file into `assets/MCB`. 
The resulting `assets/MCB` should contain a folder called `dataset_org_norm` along with `train_catalog.txt` and `test_catalog.txt` which are lists of meshes after filtering (as described in the appendix, we filter out meshes with more than 5000 triangles and keep up to 100 meshes per category).
Then follow the `script.py` in `tests/symmetry_detection` to run various methods.

To run the object detection experiment (Section 5.5 in the paper), we use fiftyone library to fetch COCO17 dataset. Be aware that running the commands in `tests/objdetect/script.py` will first download COCO17 dataset which can be huge (around 40GB, and this only happens one time).


## Extending to custom problems
To apply the proximal operator learning (POL) framework to a custom problem, you will need to
- inherit a problem class from ProblemBase and implement the abstract methods;
- create a folder in `tests/`, and then create a config Python file that suits your need.

An example is the spring equilibrium problem defined by the problem class `pol/problems/spring_equilibrium.py`, with a config file `tests/spring_equilibrium/config.py`.
These two files (and the `ProblemBase` class) are documented.

To run it, `cd` into `tests/spring_equilibrium` and then call `python config.py`.
To visualize the results, start a Jupyter lab at `notebooks/spring`. The `plot.ipynb` inside can be run directly.

