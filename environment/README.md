# Development Environment Setup

Follow the steps.

1. Clone the repo.

   ```bash
   git clone https://github.com/theislab/tardis
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
   > All mamba commands can be replaced by `conda`, not recommended though.

   - Hava a look at the guide: https://bioinformatics_core.ascgitlab.helmholtz-muenchen.de/it_hpc_documentation/Installations.html#MambaInstallation
   - Use the following versions instead of the latest,
     - for Linux: https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-23.11.0-0-Linux-x86_64.sh
     - for MAC Silicon: https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-23.11.0-0-MacOSX-arm64.sh
   - After installation `.bashrc` put,
     ```bash
     export PATH=$PATH:/home/icb/kemal.inecik/tools/apps/mamba/bin
     ```
   - Then run
     ```bash
     conda init "$(basename "${SHELL}")"
     ```
   - For installing new environment
     ```bash
     mamba env create -f environments/fcvi_env.yaml  # For M1 Mac. No GPU support.
     ```
   - Build the jupter extensions.
     ```bash
     jupyter lab build
     ```

   Currently, whether the GPU version of pytorch is installed depends on which node you install the environment on. That means that if you want to your code to reconnise GPUs when working on a cluster, please make sure you install the conda environments from a node that has access to a GPU. Test whether you installed GPU version using the following
   in Python console.

   ```python
   import torch
   assert torch.cuda.is_available()
   ```

   In case you already have an environment installed that doesn't recognise the GPU on a GPU node, remove the environment name manually and then call above command line again.

   ```bash
   mamba env remove -n tardis_env
   mamba clean -avvvy
   # rm -rf .conda .jupyter .ipython .local .netrc .npm .nv .python_history .virtual_documents .yarn .config/matplotlib
   ```

4. Create a branch `git checkout -b your_branch`

5. Open JupyterLab. Go to the Settings menu at the top. Select 'Settings Editor'.
   - 'Language Server Protocol' settings:
     - bla
   - 'Code Formatter' settings:
     - make line lenght 120 (or locate flake8 black configs)
