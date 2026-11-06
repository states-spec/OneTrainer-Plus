# Quick Start Guide

This Guide is intended to explain the basic principles of fine-tuning a model with OneTrainer in the order you will see
them in the UI. For more in-depth explanations, please check the wiki here: https://github.com/Nerogar/OneTrainer/wiki

### Preparing a Dataset

Place the images you want to train on in any directory you want. For best results, you should add prompts to every
image. There are two options:

1. For each image, add a .txt file with the same name in the same directory. The text file should contain the prompt to
   train on. i.e image1.jpg and image1.txt
2. Rename all images with the prompt you want to train on.

### Presets

To quickly get started, OneTrainer provides a set of presets for the most useful configurations. Select the one that
best
fits your needs, then change settings as you need. You can also add your own presets.

### Workspaces

OneTrainer uses workspaces to separate your training runs. A workspace contains all the backup, sampling and TensorBoard
data of a single run. When you start a new training run, it is recommended that you select an empty directory as your
workspace.

### Caching

Caching speeds up the training, by saving intermediate data from your training images on disc. This data only needs to
be created once, and can be re-used between epochs or even different training runs. If you change any data related
settings, you should either clear the cache directory, or choose a new one.

### Tensorboard

TensorBoard enables easy tracking of the training progress during training. To use it, enable Tensorboard. During the
training run, click on the TensorBoard button at the bottom to open the web interface. This web interface will display
loss values and samples.

### Input Model Settings

Select a base model to train on. This can be a model in many different formats:

1. A filename of a checkpoint either in ckpt or safetensors format
2. A directory name of a model in diffusers format
3. A Hugging Face repository name
4. A backup directory from a previous training run

You also need to specify the type of the input model.

### Output Model Settings

Simply specify the file or directory name where you want to save the final model, and the output format.
The data type defines the precision for the final model. Float16 will create smaller models, but the quality can be
reduced.

### Data Settings

Aspect Ratio Bucketing makes it possible to train the model on images with different aspect ratios at the same time. All
images are resized to roughly match the same total pixel count.

### Latent caching

Latent caching (as described in the caching section) will speed up the training by saving intermediate data. If this
setting is enabled, some data will be calculated based on your training images and saved to disc. If you enable data
augmentation, you should increase the "Image Variations" setting of that concept, otherwise only a single augmentation
version will be cached.

### Adding concepts.

Concepts define the data you want to train on. You can first create a configuration. Then you can add as many concepts
to that configuration as you want.

After adding a concept, you can click on the new widget that appears to open the
details window. In that window you can optionally select a name, then specify the directory where the images are saved.
The prompt source defines where the prompts are loaded from. You can choose to load them from `.txt` files in the same
directory, from the image filenames, or from a single `.txt` file that contains multiple lines to randomly choose from. If
you choose to use a single `.txt` file, you also need to specify where that file is saved. Optionally, you can also
include subdirectories of the selected directory in the concept definition.

On the augmentation tab, you can define how OneTrainer should modify your images before training on them. These
modifications can increase the diversity of your training data.

### Training Settings

Here you can set all the parameters that are used during training. You probably won't need to change most of these
settings, but here are some examples of the most important ones:

- Learning Rate: This setting defines how quickly you want the model to learn. But setting it too high will break the
  model.
- Epochs: This defines how many steps the model should be trained on each image in the dataset.
- Batch Size: Defines the number of images to train at once. A higher setting will lead to more generalized results, but
  will take longer to train and requires more VRAM.
- Accumulation Steps: If you can't increase the Batch Size because of VRAM constraints, you can increase this setting to
  create the same effect, but slow down the training even more.

### Training Seed

Everything random in a training run comes from one number, the training seed: where the run starts (the LoRA or
embedding initialization), its dropout, the noise and timesteps of every step, and stochastic rounding. With the same
seed and settings, two runs train alike (bit-identical in our tests on the CPU and on an RX 7900 XTX), so a change in
one setting shows its own effect instead of being mixed with a different random start. Set it in the training tab:

- Fixed (the default, seed 42): the seed in the Seed field, for every run. Use it to compare settings on the same seed.
- Increment / Decrement: trains with the seed shown, then moves it one up or down for the next run, to try a few seeds
  in a row.
- Random: one new seed for every run, drawn once and written into the Seed field before the run starts; the whole run
  comes from that one seed. To repeat a run you liked, switch to Fixed with that seed.

Compile transformer blocks (model tab) changes the rounding, so keep it the same in runs you compare. On a GPU, two
compiled runs with the same seed also end slightly apart (2.1e-3 on an RX 7900 XTX, where two seeds are about 0.8 apart),
because PyTorch picks its compiled kernels by timing them; `TORCHINDUCTOR_DETERMINISTIC=1` skips that timing (see
[LAUNCH-SCRIPTS.md](../LAUNCH-SCRIPTS.md) for how to check whether it makes compiled runs repeat exactly on your GPU).

The seed a run used is printed when it starts ("Training seed: ...") and saved with the run's config in the workspace
(`workspace/config/<run name>.json`). The data order and augmentations follow each concept's own seed (concepts tab),
and validation keeps its own fixed seed. From the command line, Increment and Decrement can't change the config file, so
`scripts/train.py` trains with the value in the file; Random picks a new seed there too. To continue a backup on the
same noise, use Fixed with the seed of the run that made it.

"Train Text Encoder" and "Train UNet" can be deactivated to only train parts of the model. In most cases, both should
stay active. You can also specify how many epochs each part should be trained for, as well as a specific learning rate
for each part.

### Train Data Type

Internally, this sets the mixed precision data type when doing the forward pass through the model. This setting trades
precision for speed during training. Not all data types are supported on all GPUs. In practice, float16 only slightly
reduces the quality, while providing a significant speed boost.

### Resolution

Your training images will be resized to that resolution when training. You don't need to do any manual resizing. You can
specify multiple resolutions as a comma-separated list. All resolutions will be trained at the same time.

### Masked training

With masked training, you can instruct the model to only learn from some parts of your training images. For example, if
you want to train a subject, but not the background, this setting will help. To enable masked training, add a mask for
your training images. This mask is a black-and-white image, where white regions define what should be included, and
black regions are excluded. The files need the same name as the images, with an added `-masklabel.png`
extension.

- The mask of `photo.jpg` is `photo-masklabel.png` in the same folder. Masks are always PNG files, also for `.jpg` or
  `.webp` images, and the name can't be changed. A file with any other name is not used as a mask (`photo-mask.png` is
  even trained as an image of its own).
- An image without a mask is trained on the whole image, the same as an all-white mask. At the start of training,
  OneTrainer says how many images of each concept have no mask and names a few of them.
- Gray mask values work as partial weights.

On every model the mask weights the loss:

- *Unmasked Weight* is the loss weight of the black areas, on every step (0: ignored, 1: same as no mask).
- *Normalize Masked Area Loss* divides each sample's loss by its mask's average weight, so samples with small masks are
  not learned more weakly.
- *Masked Prior Preservation Weight* (LoRA only) trains the black areas towards the prediction of the original model;
  set a low Unmasked Weight with it.

**Inpainting models** (Stable Diffusion 1.5/2.0 inpainting, SDXL inpainting, Flux Fill) also take the mask and the image
with the masked (white) area filled with gray as inputs, and the training tab shows two more settings for them:

- *Unmasked Probability* is the share of samples that get an all-white mask and an all-gray image instead, so the model
  also learns to generate whole images (the loss then covers the whole image).
- *Custom Conditioning Image* replaces that grayed-out image with your own `photo-condlabel.png`, for example the
  "before" image of a before/after pair to teach object removal.

Upstream OneTrainer and older versions of this fork leave the mask out of the loss of inpainting models (the loss covers
the whole image). To train one that way, set Unmasked Weight to 1.

### Sampling

Here you can define prompts to regularly sample the model during training. The results will be displayed in tensorboard
and also saved in the workspace directory.

### Backup and Continue

On the backup tab, you can configure regular backups that will be created during the training run. These backups are
saved in the workspace.

Additionally, you can decide to create a backup at the end of the training run. They
contain all the model data to continue the training run. To continue the training from a previous backup, just select
the backup directory as your base model.

A backup can be converted to a model file by using the model convert tool.

Note: Backups are **NOT** to be used externally in tools like Forge, SwarmUI etc they will not work. Utilise saves.
