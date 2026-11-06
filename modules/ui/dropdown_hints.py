# Tooltip texts that explain a dropdown's choices, shared by the views that show the same dropdown.

WEIGHT_DTYPE = (
    "float32: full precision, the most memory. "
    "bfloat16 / float16: half the memory; bfloat16 is the usual choice. "
    "float8 (W8): weights stored in 8 bits and computed in the train data type; saves memory, not faster. "
    "nfloat4: 4-bit NormalFloat weights (bitsandbytes); the least memory and precision. "
    "int W8A8: int8 weights and activations (int8 matrix math). "
    "float W8A8: float8 weights and activations; needs fp8 hardware (NVIDIA sm89+, AMD gfx942/gfx950/RX 9000). "
    "compressed: also compressed with nvCOMP (NVIDIA only). "
    "bfloat16 A8 int / float: bfloat16 weights (trainable, also in a full fine-tune), quantized with the activations "
    "to 8 bits for each matrix multiplication; A8 float needs fp8 hardware like float W8A8. "
    "GGUF: keeps the quantization of a .gguf model file; GGUF A8 also quantizes the activations. "
    "Not every part offers every choice."
)

RUN_NAME_MODE = (
    "Default: <training method>_<date and time>, new for each run. "
    "Friendly: two random words, new for each run. "
    "Custom: the name in the field, kept for every run."
)

TRAINING_SEED_MODE = (
    "Off: as before, a new initialization every run and the same noise for every run. "
    "Fixed: the seed below, every run, so runs that differ only in their settings start and step alike. "
    "Increment / Decrement: trains with the seed below, then moves it one up / down for the next run. "
    "Random: a new seed for each run, written into the field so the run can be repeated with Fixed. "
    "The seed sets the LoRA/embedding initialization, dropout and each step's noise and timesteps; the data order "
    "and augmentations follow each concept's own seed (concepts tab). Validation keeps its own fixed seed."
)

OUTPUT_DTYPE = (
    "Precision to use when saving the output model. "
    "bfloat16: the range of float32 at half the size, the usual choice. "
    "float16: finer steps than bfloat16 but a small range. "
    "float32: full precision, twice the size. "
    "float8 / nfloat4: smaller files with less precision; check that your inference tool loads them."
)

OUTPUT_FORMAT = (
    "Format to use when saving the output model. "
    "LoRA: Comfy for ComfyUI, Kohya for kohya-ss style tools, Original for the model's original key names, "
    "Diffusers for diffusers/PEFT, Legacy for the older OneTrainer layout. Comfy, Kohya and Original train "
    "fused q/k/v adapters, Diffusers and Legacy split ones, so a LoRA or backup resumes only in a format with the "
    "same layout. "
    "Full model: Diffusers (a folder), Original (single file), or the transformer only with original or Comfy names. "
    "Safetensors (embedding training): the embedding file."
)

PEFT_TYPE = (
    "The type of low-parameter finetuning method. "
    "LoRA: two low-rank matrices per layer, optionally with DoRA (weight decomposition). "
    "LoHa: the Hadamard product of two low-rank pairs; more expressive at the same rank. "
    "LoKr: a Kronecker product; very compact, optionally with DoRA. "
    "OFT v2: a learned block-wise rotation of the weights, which keeps their norms (DoRA OFT adds a learned scale per output channel)."
)

LEARNING_RATE_SCHEDULER = (
    "Learning rate scheduler that automatically changes the learning rate during training. "
    "CONSTANT: fixed. LINEAR: decreases linearly to the minimum factor. COSINE: cosine decay. "
    "COSINE_WITH_RESTARTS: cosine waves down and back up (Learning Rate Cycles). "
    "COSINE_WITH_HARD_RESTARTS: decays, then jumps back to the full rate each cycle. "
    "REX: decays slowly first and fast at the end. "
    "ADAFACTOR: Adafactor's own relative step size (ADAFACTOR optimizer). "
    "CUSTOM: your own scheduler class (the ... button). "
    "Schedule-free optimizers always run CONSTANT."
)

TIMESTEP_DISTRIBUTION = (
    "Selects the function to sample timesteps during training. "
    "UNIFORM: every noise level equally often. "
    "LOGIT_NORMAL: centered on the middle noise levels (SD3); Noising Bias shifts it, Noising Weight widens it. "
    "HEAVY_TAIL / COS_MAP: the SD3 paper's heavy-tailed and cosine-mapped variants. "
    "SIGMOID / INVERTED_PARABOLA: weighted by Noising Weight and Noising Bias. "
    "BETA: a Beta distribution with alpha = Noising Weight, beta = Noising Bias. "
    "Use the preview to see the result."
)

LOSS_WEIGHT_FUNCTION = (
    "Choice of loss weight function. Can help the model learn details more accurately. "
    "CONSTANT: all timesteps weigh the same. "
    "SIGMA (flow matching models): weighs each sample by its noise level, more weight on noisier steps. "
    "MIN_SNR_GAMMA (diffusion models): caps the weight of low-noise steps (strength = gamma, often 5). "
    "P2: weighs down low-noise steps by (1 + SNR)^-strength. "
    "DEBIASED_ESTIMATION: weighs by 1 / sqrt(SNR). "
    "Only the functions the model's loss supports are listed."
)

EMA_MODE = (
    "EMA averages the training progress over many steps, better preserving different concepts in big datasets. "
    "OFF: no EMA. GPU: the EMA weights stay on the GPU (fast, uses VRAM). "
    "CPU: they stay on the CPU (saves VRAM, slower updates). Saved models contain the EMA weights."
)

TRAIN_DTYPE = (
    "The mixed precision data type used for training. This can increase training speed, but reduces precision. "
    "bfloat16: the usual choice on recent GPUs. float16: needs loss scaling and overflows more easily. "
    "float32: full precision, slow. tfloat32: float32 with TF32 matrix math where the GPU has it (NVIDIA Ampere+)."
)

SAMPLER = (
    "The sampler for the sample images. Euler A and DPM++ SDE add fresh noise each step (more variety); the others "
    "are deterministic. The Karras variants use the Karras noise schedule, with more steps at low noise."
)

SVD_DTYPE = (
    "What datatype to use for SVDQuant weights decomposition. "
    "disabled: off. float32 / bfloat16: the data type of the low-rank part SVDQuant keeps beside the quantized "
    "weights, which absorbs their largest errors."
)
