# OneTrainer Launch Scripts


## Mac and Linux Systems

### The launch system consists of the following scripts:

- `install.sh`: Ensures that you have a valid Python runtime environment and installs all requirements if necessary.
- `update.sh`: Updates OneTrainer to the latest version and upgrades any outdated requirements in your Python runtime environment.
- `start-ui.sh`: Launches the main OneTrainer interface.
- `run-cmd.sh`: Executes a custom script (such as "train"), and supports providing command-line arguments. See the "running custom script commands" guide section for more details.


### All of the scripts accept the following *optional* environment variables to customize their behavior:

- `OT_CONDA_CMD`: Sets a custom Conda command or an absolute path to the binary (useful when it isn't in the user's `PATH`). If nothing is provided, we detect and use `CONDA_EXE` if available, which is a variable that's set by Conda itself and always points at the user's installed Conda binary.

- `OT_CONDA_ENV`: Sets the directory name (or an absolute/relative path) of the Conda environment. If a name or relative path is used, it will be relative to the OneTrainer directory. Defaults to `conda_env`.

- `OT_PYTHON_CMD`: Sets the Host's Python executable. It's used for creating the Python Venvs. This can be used to force the usage of a specific Python version's binary (such as `python3.11`) whenever the host has multiple versions installed. However, it's *always* recommended to use Conda or Pyenv instead, rather than relying on the host's unreliable system-wide Python binaries (which might change or be removed with system updates), so we don't recommend changing this option unless you *really* know what you're doing. Defaults to `python`.

- `OT_PYTHON_VENV`: Sets the directory name (or an absolute/relative path) of the Python Venv. If a name or relative path is used, it will be relative to the OneTrainer directory. Defaults to `venv`.

- `OT_PREFER_VENV`: If set to `true`, Conda will be ignored even if it exists on the system, and Python Venv will be used instead. This ensures that people who use `pyenv` (to choose which Python version to run on the host) can easily set up their desired Python Venv environments. Set it to `false` to use Conda when it is installed. Defaults to `true`.

- `OT_LAZY_UPDATES`: If set to `true`, OneTrainer's self-update process will only update the Python environment's dependencies if the OneTrainer source code has been modified since the previous dependency update. This speeds up executions of `update.sh`, and is generally safe, but may miss some updates and important bugfixes for external third-party dependencies. If you use this option, you must set it permanently for *every* script (not just `update.sh`). Defaults to `false`.

- `OT_CUDA_LOWMEM_MODE`: If set to `true`, it enables aggressive garbage collection in PyTorch to help with low-memory GPUs. Defaults to `false`.

- `OT_EXPANDABLE_SEGMENTS`: If set to `true`, PyTorch maps GPU memory in growable segments (`expandable_segments:True`), so memory freed by one model part (for example the transformer, while the text encoder runs for sampling) can be reused for another without fragmenting. Helps when an out-of-memory error says that much memory is "reserved by PyTorch but unallocated". Can be combined with `OT_CUDA_LOWMEM_MODE`. Linux only. Defaults to `false`, because PyTorch has no fallback: on a GPU or driver without virtual memory support every allocation fails. Check yours once before using it, from the OneTrainer folder: `env PYTORCH_ALLOC_CONF=expandable_segments:True venv/bin/python -c "import torch; x = torch.ones(2**28, device='cuda'); print('expandable segments work:', x.sum().item())"` (use `conda_env/bin/python` with Conda). It should print `expandable segments work: 268435456.0`. That only shows it allocates. On an AMD RX 7900 XTX (ROCm 7.2, torch 2.13) it gives wrong results (VAE encodes, convolution gradients, layer offloading), so don't use it there; on other AMD GPUs run `./run-cmd.sh rocm_check --only allocator` first and use it only if that check passes.

- `OT_ROCM_MEM_POOLS`: If set to `true`, layer offloading and model part moves keep their GPU memory in dedicated PyTorch MemPools on ROCm too (on NVIDIA they always do), so a part's memory goes back to the driver in one piece when it is unloaded. Defaults to `false` on ROCm, because MemPools are another allocator mode and the `expandable_segments` one gives wrong results on an RX 7900 XTX. Run `./run-cmd.sh rocm_check --only mem_pool` first and use it only if that check passes. On an RX 7900 XTX they trained a Chroma LoRA bit-identically, but saved no memory and no time, with layer offloading or without.

- `PYTORCH_TUNABLEOP_ENABLED` (PyTorch's own variable, passed through like the others): If set to `1`, PyTorch times the available matrix multiplication kernels for every new shape the first time it appears and then uses the fastest. The results are kept in `diagnostics/tunableop/tunableop_results0.csv` in the OneTrainer folder (unless `PYTORCH_TUNABLEOP_FILENAME` names another file) and reused by later runs, so only the first run with a new model, resolution, batch size or caption-length mix pays for the tuning. On an AMD RX 7900 XTX, a Chroma LoRA at 512 px trained 9% faster per step with it (tuning took about 10 minutes once) and needed about 0.4 GB more GPU memory; the tuned kernels are as accurate as the default ones, and the results differed from a run without it far less than two runs with different seeds do. Delete the file after updating PyTorch or ROCm. `./run-cmd.sh rocm_check --full --only "runtime switches"` measures it on other GPUs. Defaults to off.

- `OT_LOG_VRAM`: If set to `true`, OneTrainer prints the GPU memory use every time it loads or unloads a model part (caching, sampling, backups), to find the step that runs out of memory. Defaults to `false`.

- `OT_DIAG_LOG`: If set to `true`, a training run writes a diagnostics log to `diagnostics/runs/<timestamp>.jsonl` in the OneTrainer folder (any other value is the file to write): GPU memory, fragmentation, allocation retries and timing at every step, part load/unload, caching, sampling, validation, backup and save, and the error that ended a run. `./run-cmd.sh diag_report` summarizes and compares such logs, and `./run-cmd.sh diag_ab` runs a config under several settings (for example with and without `OT_ROCM_MEM_POOLS`) and compares them. See [docs/RunDiagnostics.md](docs/RunDiagnostics.md). Defaults to off.

- `OT_SEED`: If set to a number, the random generators are seeded with it when training starts, so two runs of the same config start from the same LoRA (or embedding) initialization, which is otherwise random each run. For comparing runs; `diag_ab` sets it. Defaults to unset.

- `OT_DETERMINISTIC`: If set to `true`, PyTorch is asked for deterministic GPU kernels, so two runs of the same config differ less (operations without a deterministic kernel print a warning and run as before). For comparing runs; can be slower. Defaults to `false`.

- `OT_PLATFORM_REQUIREMENTS`: Allows you to override which platform-specific "requirements" file you want to install. Defaults to `detect`, which automatically detects whether you have an AMD or NVIDIA GPU. But people with multi-GPU systems can use this setting to force a specific GPU acceleration framework's requirements. Valid values are `requirements-rocm.txt` for AMD, `requirements-cuda.txt` for NVIDIA, and `requirements-default.txt` for non-AMD/NVIDIA systems. A path to another file also works, e.g. to compare PyTorch/ROCm builds in a separate copy of OneTrainer (see [docs/RocmVersionComparison.md](docs/RocmVersionComparison.md)); pass it to `update.sh` there too.

- `OT_SCRIPT_DEBUG`: If set to `true`, it enables additional debug logging in the scripts. Defaults to `false`.


### Examples of how to use the custom environment variables:

- You can provide custom environment variables directly on the command line, as follows: `env OT_PREFER_VENV="true" OT_CUDA_LOWMEM_MODE="true" OT_PLATFORM_REQUIREMENTS="requirements-cuda.txt" ./start-ui.sh`.
- You can add them to your user's persistent environment variables, so that they are always active. The process varies depending on your operating system. On Linux, you can place them in `~/.config/environment.d/onetrainer.conf` (on all Systemd-based distros), which is a plaintext file with *one variable per line,* such as `OT_CUDA_LOWMEM_MODE="true"`. Beware that changes to `environment.d` requires a *complete system restart* to take effect (there is no command for reloading them live). To verify that your environment has been set persistently, you can then open a terminal window and run `printenv <variable name>` (such as `printenv OT_CUDA_LOWMEM_MODE`) to see if your custom values have taken effect.
- If you're launching OneTrainer from your own, custom scripts, then you can instead `export` the new values (which tells the shell to pass those environment variables onto child processes). For example, by having a line such as `export OT_CUDA_LOWMEM_MODE="true"` before your script calls `./OneTrainer/start-ui.sh`.
- If you're running OneTrainer inside a Docker/Podman container, you can instead use the [ENV](https://docs.docker.com/reference/dockerfile/#env) instruction in your `Dockerfile` / `Containerfile` to set the variables, such as `ENV OT_CUDA_LOWMEM_MODE="true"`.


### Installing the required Python version for OneTrainer:

- If you've received a warning that your system's Python version is incorrect, then your system most likely doesn't have Conda installed, and has instead tried to create a Python Venv with your host's default Python version. If that version is incompatible with OneTrainer, then you'll have to resolve the problem by manually installing a compatible version. Alternatively, you are using an outdated Conda environment.
- Begin by deleting the `venv` sub-directory inside the OneTrainer directory, to erase the invalid Python Venv (which was created with the wrong Python version). If you were using Conda, then you must instead delete the outdated `conda_env` sub-directory.
- Now you'll have to choose which solution you prefer.
- The most beginner-friendly solution is to install [Miniconda](https://docs.anaconda.com/miniconda/) on your system and set `OT_PREFER_VENV="false"` (see above; Python Venv is preferred by default). OneTrainer will then automatically install and manage the correct Python version for you via Conda. You can stop reading here if you're choosing this solution. Everything will work automatically after that.
- Alternatively, if you prefer a more lightweight and advanced solution, then you can use [pyenv](https://github.com/pyenv/pyenv), which allows you to set the exact Python version to use for OneTrainer's directory. If you're on Linux, then read their "[automatic installer](https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer)" section and follow the instructions. If you're on a Mac instead, then read their "[Homebrew](https://github.com/pyenv/pyenv?tab=readme-ov-file#homebrew-in-macos)" section (which is an open-source package manager for Macs).
- After installing pyenv, you will also need to install the [Python build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) on your system, since pyenv installs each Python version by compiling them directly from the official source code.
- Restart your shell, and then try the `pyenv doctor` command, which ensures that pyenv is loaded and verifies that your system contains all required dependencies for installing Python.
- Run `pyenv install <python version>` to install whichever Python version is currently required by OneTrainer. You can look at the `OT_CONDA_USE_PYTHON_VERSION` variable at the top of the `lib.include.sh` file in OneTrainer's project directory, to see which Python version is recommended by OneTrainer at the moment.
- Lastly, you must navigate to the OneTrainer directory, and then run `pyenv local <python version>` to force OneTrainer to use that version of Python. Your choice will be stored persistently in the hidden `.python-version` file, and can be changed again in the future by running the command again.
- You can now run `python --version` to verify that the `python` command in OneTrainer is being mapped to the correct Python version by pyenv.
- Everything is now ready for running OneTrainer!


### Running custom script commands:

- Always use `run-cmd.sh` when you want to execute any of OneTrainer's CLI tasks. It automatically validates the chosen target script's name, configures the runtime environment correctly, and then runs the target script with your given command-line arguments.
- For example, to run the training CLI script, you would use `./run-cmd.sh train --config-path <path to your config>`.
- The names of all valid scripts can be seen in OneTrainer's `scripts/` sub-directory.
- To learn more about the available command-line arguments for each script, you can execute them with the `-h` (help) argument: `./run-cmd.sh <script name> -h`. For example, if you want to learn more about the "train" script, you would run `./run-cmd.sh train -h`.


### Creating your own launch scripts and automating tasks:

- If you want to automate various OneTrainer CLI tasks, then you should call `run-cmd.sh` from your own scripts (see previous guide section), since it's capable of running *any* OneTrainer command with your own command-line arguments.
- To run multiple tasks in the same scripts, you should perform separate calls to `run-cmd.sh`. Run it as many times as required for all the custom scripts and command-line arguments that you want to perform in your own script.
- It's highly recommended that you use `set -e` at the top of your own scripts (see `install.sh` for an example of that), since it tells Bash to exit your script if any of the OneTrainer commands fail. Otherwise your script will continue running even if a previous step has failed, which is usually not what you want!
