# Development Environment Setup

Follow the steps.

1. Clone the repo.

   ```bash
   git clone https://github.com/theislab/cannot
   ```

2. Necessary to run beforehand for MAC Silicon:

   > Note: Assumes that [homebrew](https://brew.sh) was already installed.

   ```bash
   brew install openblas
   brew install pyenv --head
   # Make sure package manager can find these paths. Export paths before the installation.
   export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
   export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
   export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
   ```

3. Install the environment.

   > Note: [mamba](https://github.com/conda-forge/miniforge) (mambaforge) improves install times drastically.
   > All mamba commands can be replaced by `conda`.

   ```bash
   mamba env create -f environment/cannot_env.yaml  # For M1 Mac. No GPU support.
   ```

   Currently, whether the GPU version of pytorch is installed depends on which node you install the environment on. That means that if you want to your code to reconnise GPUs when working on a cluster, please make sure you install the conda environments from a node that has access to a GPU. Test whether you installed GPU version using the following
   in Python console.

   ```python
   import torch
   assert torch.cuda.is_available()
   ```

   In case you already have an environment installed that doesn't recognise the GPU on a GPU node, remove the environment name manually and then call above command line again.

   ```bash
   mamba env remove -n cannot_env
   mamba clean -avvvy
   ```

4. Create a branch branch `git checkout -b your_branch`

5. Build the jupter extensions.

   ```bash
   jupyter nbextension enable --py widgetsnbextension
   jupyter lab build
   ```

6. Open `training_demo.ipynb` under `notebooks` directory, and a preprocessing file such as `he2022.ipynb` under `preprocessing` and make sure you can run all of the cells.
