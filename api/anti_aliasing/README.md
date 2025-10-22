### What is Aliasing?

Aliasing is a distortion that occurs when a signal is sampled at a rate too low to accurately capture its high-frequency components.
According to the [Nyquist–Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem), a signal must be sampled at least **twice its highest frequency to preserve its information**.
When this condition isn’t met, high-frequency content “folds back” into the lower-frequency spectrum — creating false or misleading frequency components known as aliases.
In audio, this can result in unnatural tones, harsh distortions, or loss of clarity.

To mitigate these artifacts, the model introduces a hybrid anti-aliasing approach that combines traditional DSP filtering with neural-based refinement:

Classical DSP Stage:
A low-pass filter (e.g., FIR or Butterworth) is applied before downsampling to remove frequency components above the Nyquist limit — preventing them from causing aliasing.

Neural Enhancement Stage:
A lightweight neural network (e.g., CNN or small feedforward model) learns to reconstruct and smooth signal features lost during filtering.
It refines the output by modeling subtle nonlinear relationships that classical filters may overlook.

Adaptive Generalization:
The model is evaluated across different sampling frequencies, showing its ability to maintain consistent accuracy and clarity, even when the signal bandwidth changes.