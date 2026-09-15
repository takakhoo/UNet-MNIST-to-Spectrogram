# U-Net: Handwritten Digits to Musical Spectrograms

A cross-modal deep-learning experiment that maps a 28×28 MNIST/EMNIST digit to
a 1008×1008 spectrogram representing a major triad. The project combines a
procedural target-generation pipeline, a PyTorch U-Net, perceptual image losses,
and waveform synthesis.

## Project idea

Digits `0`–`7` stand in for scale degrees. For each input image, the data
pipeline creates a target spectrogram whose vertical structure encodes a
fundamental, major third, and perfect fifth. A U-Net then learns the mapping
while preserving the digit's spatial character through skip connections.

## Technical highlights

- Procedural supervision built with resampling, Gaussian decay and smearing,
  harmonic row selection, and Perlin-noise texture
- Encoder–decoder model with skip connections and a 1008×1008 output
- Huber and multi-scale SSIM objectives for pixel and structural fidelity
- Mixed-precision training and checkpointing in PyTorch
- Spectrogram-to-waveform synthesis for audible qualitative evaluation

## Quick start

```bash
git clone https://github.com/takakhoo/UNet-MNIST-to-Spectrogram.git
cd UNet-MNIST-to-Spectrogram
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab UNet_MNIST_to_Spectrogram.ipynb
```

The committed checkpoint is an interrupted training snapshot and may require
matching model dimensions. Start with the executed notebook to understand the
data pipeline before resuming training.

## Artifacts

- `UNet_MNIST_to_Spectrogram.ipynb` — complete experiment notebook
- `Paper MNIST to Chord Spectrogram.pdf` — project report
- `Final Presentation.pptx` — presentation deck
- `thenumber*.png` / `thenumber*.npy` — conditioning-stage examples
- `redu_U-Nettttttt_checkpoint_step_156819_INTERRUPTED.pth` — historical
  checkpoint
- `Upscaler/` — Real-ESRGAN exploration and model assets

## Contributors

Taka Khoo, Harry Leiter, and Doruk Ozel developed this project for ENGS 106.

## Scope

This is a research/course prototype. The audio examples provide qualitative
evidence; a stronger follow-up would add held-out metrics, deterministic
training configuration, and a lightweight inference script.
