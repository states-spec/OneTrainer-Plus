# OneTrainer+

OneTrainer+ (OneTrainer-Plus) is a fork of [OneTrainer](https://github.com/Nerogar/OneTrainer) by Nerogar, a one-stop
solution for all your Diffusion training needs. It keeps OneTrainer's configs, presets and model files, and adds fixes,
features ported from [Koratahiu's OneTrainer fork](https://github.com/Koratahiu/OneTrainer) and from upstream's
preview-2 branch, testing on AMD GPUs with ROCm, and tools to check a GPU setup and compare training runs. The people
whose work it is built on are listed under [Credits](#credits).

<a href="https://discord.gg/KwgcQd5scF"><img src="https://discord.com/api/guilds/1102003518203756564/widget.png" alt="OneTrainer Discord"/></a> (OneTrainer's community Discord)<br>

## What OneTrainer+ adds

-   **Fixes**: training, saving and resuming problems found and fixed in this fork, among them optimizer settings with a
    fused back pass, EMA weights in intermediate saves, NaN losses from masked training with an empty area, timesteps
    outside the noise schedule, LoRA files with layers outside the layer filter, and resuming from unfinished or
    mismatched backups. Settings that can't work together are greyed out or stopped before the model loads.
-   **AMD GPUs (ROCm)**: tested on a Radeon RX 7900 XTX with ROCm 7.2. Among others, attention with head size 512 (every
    VAE) runs on a correct kernel on RDNA3, and LARS works on bitsandbytes 0.49.2. `rocm_check` checks and measures a GPU
    setup ([docs/GpuCheck.md](docs/GpuCheck.md)); [docs/RocmVersionComparison.md](docs/RocmVersionComparison.md) compares
    PyTorch/ROCm builds.
-   **Optimizers**: Advanced_Optimizers (`adv_optm`) 2.6.0 with its 2.5 and 2.6 options (Scaled Weight Decay among
    them), the parameter tags its optimizers need, and resumes that keep Prodigy's step size.
-   **Training options**: CIOP, CEP and Immiscible Diffusion noise, an exact OFT rotation (CANS), DoRA-OFT and OFT
    spectral norm clipping (all off by default), and stochastic rounding for bfloat16 EMA weights (on by default).
-   **Cloud training** installs OneTrainer+ from this repository on the cloud machine (upstream's default installed
    OneTrainer; configs that still name upstream's default repository and folder switch to this one when loaded).
-   **Diagnostics**: a JSONL record of each run, run comparisons and A/B runs, a per-step optimizer view and a config
    diff ([docs/RunDiagnostics.md](docs/RunDiagnostics.md)).
-   **UI**: a dark theme by default, a settings tab (theme, scale, font size), tooltips that explain each choice, and
    settings shown only where the model and training method use them.

## Features

-   **Supported models**: Ernie Image, Z-Image, Qwen Image, FLUX.1, Flux.2 Dev and Klein, Chroma, Stable Diffusion 1.5, 2.0, 2.1, 3.0, 3.5, SDXL, Würstchen-v2, Stable Cascade,
    PixArt-Alpha, PixArt-Sigma, Sana, Hunyuan Video and inpainting models
-   **Model formats**: diffusers and ckpt models
-   **Training methods**: Full fine-tuning, LoRA, embeddings
-   **Masked Training**: Let the training focus on just certain parts of the samples
-   **Automatic backups**: Fully back up your training progress regularly during training. This includes all information to seamlessly continue training
-   **Image augmentation**: Apply random transforms such as rotation, brightness, contrast or saturation to each image sample to quickly create a more diverse dataset
-   **TensorBoard**: A simple TensorBoard integration to track the training progress
-   **Multiple prompts per image**: Train the model on multiple different prompts per image sample
-   **Noise Scheduler Rescaling**: From the paper
    [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)
-   **EMA**: Train your own EMA model. Optionally keep EMA weights in CPU memory to reduce VRAM usage
-   **Aspect Ratio Bucketing**: Automatically train on multiple aspect ratios at a time. Just select the target resolutions, buckets are created automatically
-   **Multi-Resolution Training**: Train multiple resolutions at the same time
-   **Dataset Tooling**: Automatically caption your dataset using BLIP, BLIP2 and WD-1.4, or create masks for masked training using ClipSeg or Rembg
-   **Model Tooling**: Convert between different model formats from a simple UI
-   **Sampling UI**: Sample the model during training without switching to a different application

![OneTrainerGUI.gif](resources/images/OneTrainerGUI.gif)

> [!NOTE]
> Explore OneTrainer's 📚 wiki for essential tips and tutorials after installing. Start [here!](https://github.com/Nerogar/OneTrainer/wiki).
> For command-line usage, see the [CLI Mode section](#cli-mode).


## Installation

> [!IMPORTANT]
> Installing OneTrainer+ requires Python >=3.11 and <3.14.
> You can download Python at https://www.python.org/downloads/windows/.
> Then follow the below steps.

#### Automatic installation

1. Clone the repository `git clone https://github.com/states-spec/OneTrainer-Plus.git`
2. Run:
    - Windows: Double click or execute `install.bat`
    - Linux and Mac: Execute `install.sh`

#### Manual installation

1. Clone the repository `git clone https://github.com/states-spec/OneTrainer-Plus.git`
2. Navigate into the cloned directory `cd OneTrainer-Plus`
3. Set up a virtual environment `python -m venv venv`
4. Activate the new venv:
    - Windows: `venv\scripts\activate`
    - Linux and Mac: Depends on your shell, activate the venv accordingly
5. Install the requirements `pip install -r requirements.txt`

> [!Tip]
> Some Linux distributions are missing required packages for instance: On Ubuntu you must install `libGL`:
>
> ```bash
> sudo apt-get update
> sudo apt-get install libgl1
> ```
>
> Additionally it's been reported Alpine, Arch and Xubuntu Linux may be missing `tkinter`. Install it via `apk add py3-tk` for Alpine and `sudo pacman -S tk` for Arch.

## Updating

#### Automatic update

-   Run `update.bat` or `update.sh`

#### Manual update

1. Cd to folder containing the repo `cd OneTrainer-Plus`
2. Pull changes `git pull`
3. Activate the venv `venv/scripts/activate`
4. Re-install all requirements `pip install -r requirements.txt --force-reinstall`

## Usage

OneTrainer+ can be used in **two primary modes**: a graphical user interface (GUI) and a **command-line interface (CLI)** for finer control.

For a technically focused quick start, see the [Quick Start Guide](docs/QuickStartGuide.md) and for a broader overview, see the [Overview documentation](docs/Overview.md). Otherwise visit [OneTrainer's wiki!](https://github.com/Nerogar/OneTrainer/wiki)

### GUI Mode

#### Windows

-   To start the UI, navigate to the OneTrainer-Plus folder and double-click `start-ui.bat`

#### Unix-based systems

-   Execute `start-ui.sh` and the GUI will pop up.

### CLI Mode

If you need more control or a headless approach OneTrainer+ also supports the command-line interface. All commands **need** to be run inside the active venv created during installation.

All functionality is split into different scripts located in the `scripts` directory. This currently includes:

-   `train.py` The central training script
-   `train_ui.py` A UI for training
-   `caption_ui.py` A UI for manual or automatic captioning and mask creation for masked training
-   `convert_model_ui.py` A UI for model conversions
-   `convert_model.py` A utility to convert between different model formats
-   `sample.py` A utility to sample any model
-   `create_train_files.py` A utility to create files needed when training only from the CLI
-   `generate_captions.py` A utility to automatically create captions for your dataset
-   `generate_masks.py` A utility to automatically create masks for your dataset
-   `calculate_loss.py` A utility to calculate the training loss of every image in your dataset
-   `rocm_check.py` Checks and measures the GPU setup (ROCm or CUDA): kernel correctness, the training code on the GPU, and speed per model family. See [docs/GpuCheck.md](docs/GpuCheck.md)

To learn more about the different parameters, execute `<script-name> -h`. For example `python scripts\train.py -h`

If you are on Mac or Linux, you can also read [the launch script documentation](LAUNCH-SCRIPTS.md) for detailed information about how to run OneTrainer+ and its various scripts on your system.

## Troubleshooting

For general questions about OneTrainer, ask in its [Discussions](https://github.com/Nerogar/OneTrainer/discussions), check its [Wiki](https://github.com/Nerogar/OneTrainer/wiki) or join its [Discord](https://discord.gg/KwgcQd5scF). Problems with what OneTrainer+ adds or changes belong in [this repository's issues](https://github.com/states-spec/OneTrainer-Plus/issues).

If you encounter a reproducible error you first must run update.bat or update.sh and confirm the issue is still able to be reproduced. Then export anonymized debug information to help us solve an issue you are facing and upload it as part of your Github Issues submission.

-   On Windows double click `export_debug.bat`
-   On Unix-based systems execute `./run-cmd.sh generate_debug_report`

These will both create `diagnostics/debug_report.log`.

> [!WARNING]
> Attach this file to an issue report: without it (or the same information given by hand), most issues can't be looked into.

## Contributing

Contributions are always welcome in any form. Changes that apply to OneTrainer itself are best contributed [upstream](https://github.com/Nerogar/OneTrainer) too, where every OneTrainer user gets them; for new functionality there, open a GitHub discussion or join OneTrainer's Discord to align and avoid duplicated work. You can find more information about contributing [here](docs/Contributing.md).

Before you start looking at the code, I recommend reading about the project structure [here](docs/ProjectStructure.md).
For in depth discussions, you should consider joining the [Discord](https://discord.gg/KwgcQd5scF) server.

You also **NEED** to **install the required developer dependencies** for your current user and enable the Git commit hooks, via the following commands (works on all platforms; Windows, Linux and Mac):

> [!IMPORTANT]
> Be sure to run those commands _without activating your venv or Conda environment_, since [pre-commit](https://pre-commit.com/) is supposed to be installed outside any environment.

```sh
cd OneTrainer-Plus
pip install -r requirements-dev.txt
pre-commit install
```

Now all of your commits will automatically be verified for common errors and code style issues, so that code reviewers can focus on the architecture of your changes without wasting time on style/formatting issues, thus greatly improving the chances that your pull request will be accepted quickly and effortlessly.

## Related Projects

-   **[MGDS](https://github.com/Nerogar/mgds)**: A custom dataset implementation for Pytorch that is built around the idea of a node based graph.
-   **[Stability Matrix](https://github.com/LykosAI/StabilityMatrix)**: A swiss-army knife installer which wraps and installs a broad range of diffusion software packages including OneTrainer
-   **[Visions of Chaos](https://softology.pro/voc.htm)**: A collection of machine learning tools that also includes OneTrainer.
-   **[StableTuner](https://github.com/devilismyfriend/StableTuner)**: A now defunct (archived) training application for Stable Diffusion. OneTrainer takes a lot of inspiration from StableTuner and wouldn't exist without it.

## Credits

OneTrainer+ is built on the work of many people. Numbers such as #992 are pull requests in
[Nerogar/OneTrainer](https://github.com/Nerogar/OneTrainer/pulls).

-   **OneTrainer** was created and is maintained by [Nerogar](https://github.com/Nerogar), who also wrote its data
    loader [MGDS](https://github.com/Nerogar/mgds). After Nerogar, OneTrainer's most frequent committers are dxqb, O-J1,
    SirTrippsalot, Hameer Abbasi, Morgon Kanter, Koratahiu, dxqbYD, Arcitec, aplio, prog0111, Lolzen and efhosci, with
    many more in its [contributor list](https://github.com/Nerogar/OneTrainer/graphs/contributors). The Qt (PySide6) UI
    is dxqb's work (#1445, #1446), extended by O-J1 in preview-2 (#1744).
-   **Koratahiu** wrote [Advanced_Optimizers](https://github.com/Koratahiu/Advanced_Optimizers) (`adv_optm`, the
    `*_ADV` optimizers such as ADOPT_ADV and PRODIGY_ADV) and contributed them to OneTrainer (#992), along with
    Kourkoutas-β (#1038), Huber loss (#1058), OFT v2 (#1073, #1135), Muon and AdaMuon (#1064, #1103), LoKr (#1464),
    generalized offset noise (#947), the BETA timestep distribution (#1225), scaled OFT (#1315) and, in preview-2,
    DoRA-OFT (#1335) and the scaled optimizers, centered weight decay and factored second moment (#1344). From his own
    OneTrainer fork, [Koratahiu/OneTrainer](https://github.com/Koratahiu/OneTrainer), OneTrainer+ ported:
    -   CIOP (coordinated input-output noise perturbation) and CEP (conditional embedding perturbation)
    -   Immiscible Diffusion noise sampling
    -   stochastic rounding for bfloat16 EMA weights
    -   the faster Cayley-Neumann series for OFT, the exact OFT rotation (CANS) and the OFT dtype fix on Conv2d layers
    -   the adapter parameter tags the `*_ADV` optimizers read (LoRA A/B, DoRA magnitude, OFT)
    -   the `adv_optm` 2.5 options, including SINKSGD_ADV
    -   loading Anima single-file transformers with their own number of blocks
-   **Upstream's preview-2 branch**, ported before its release, by its authors:
    -   dxqb: bfloat16 A8 weight types (#1108), the materialize/evict API (#1617), offloading fixes (#1649), chunked W8A8
        quantization (#1659), the fused LoRA epilogue (#1680), allocator diagnostics (#1619), the VAE batch-norm
        statistics on the train device (from #1608), startup and progress bar improvements (#1628, #1639) and the
        light/dark toggle (#1674)
    -   O-J1: the Hugging Face access check (#1683), simplex offloading (#1739), run names and the output directory
        (#1744) and OFT spectral norm clipping (#1751)
    -   Hameer Abbasi: arena MemPools and disk streaming (#1732)
    -   CrunchyBanana1: the preset name in the top bar (#1592)
    -   Koratahiu: DoRA-OFT (#1335) and the `adv_optm` features (#1344)
    -   Hameer Abbasi and Koratahiu: the dependency update to Pillow 13 and `adv_optm` 2.6.0, with scaled weight decay
        and SinkSGD's OFT mode (#1756)

## License

OneTrainer+ is licensed under the GNU Affero General Public License v3.0, as OneTrainer is ([LICENSE.txt](LICENSE.txt)).
It is a modified version of OneTrainer, changed since September 2026; every change is in this repository's history.
