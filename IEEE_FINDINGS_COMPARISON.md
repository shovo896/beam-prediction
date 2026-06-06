# Comparative Analysis With the Reference Study

## V. RESULTS AND DISCUSSION

### A. Experimental Comparability

The present implementation evaluates the same four sensing modalities studied
in [1]: camera image, GPS position, GPS position with drone height, and GPS
position with height and distance. Accordingly, the comparison focuses on the
top-\(k\) beam prediction accuracy, where a prediction is correct if the
ground-truth beam belongs to the model's \(k\) highest-ranked outputs.

The results should be interpreted as a comparative implementation rather than
an exact reproduction of [1]. The reference study downsampled the original
64-element received-power vector to a 32-beam classification problem and used
8,402 training and 3,602 test samples. In contrast, the current position-family
experiments use 6,832 training, 3,416 validation, and 1,139 test samples, while
the implemented output layers contain 64 or 65 units. Furthermore, [1] used a
two-hidden-layer MLP with a batch size of 32 for the position-based models,
whereas the present implementation uses three 512-unit hidden layers and a
batch size of 128. For 617 of the 11,387 position-family samples, the raw
height and distance files were unavailable; these values were imputed using
training-set medians. These differences prevent a strictly controlled
one-to-one accuracy comparison.

### B. Beam Prediction Accuracy

**TABLE I**  
**TOP-\(K\) ACCURACY OF THE PRESENT IMPLEMENTATION**

| Modality | Top-1 (%) | Top-2 (%) | Top-3 (%) | Top-5 (%) |
|---|---:|---:|---:|---:|
| Position | 57.59 | 82.00 | 91.92 | 97.81 |
| Position + Height | 69.71 | 89.03 | 95.17 | 99.03 |
| Position + Height + Distance | 74.19 | 89.99 | 96.05 | 98.95 |
| Image, refined ResNet-50 | **88.32** | **97.81** | **99.56** | **99.82** |

![Comparison of the present findings and the directly reported reference-paper results.](figures/ieee_findings_comparison.png)

**Fig. 1.** Beam prediction accuracy comparison. Panel (a) reports the complete
Top-\(k\) results of the present implementation. Panel (b) compares only the
metrics for which the reference paper provides explicit numerical values.

Table I shows that the image-based model achieves the strongest Top-1
performance, followed by the position-height-distance and position-height
models. The position-only model produces the lowest Top-1 accuracy. This
ranking is consistent with [1], which reported that vision-aided beam
prediction outperformed the position-based alternatives and that position
alone was insufficient for reliable drone beam prediction.

**TABLE II**  
**DIRECT COMPARISON WITH NUMERIC RESULTS REPORTED IN [1]**

| Metric | Reference [1] (%) | Present Work (%) | Difference (percentage points) |
|---|---:|---:|---:|
| Image Top-1 | 86.32 | **88.32** | **+2.00** |
| Image Top-3 | 99.41 | **99.56** | **+0.15** |
| Image Top-5 | 99.69 | **99.82** | **+0.13** |
| Position Top-1 | approximately 59 | 57.59 | approximately -1.41 |

The refined image model exceeds the numerical image results reported in [1]
for all directly comparable metrics. Its Top-1 accuracy improves from 86.32%
to 88.32%, corresponding to an absolute gain of 2.00 percentage points. The
Top-3 and Top-5 gains are smaller because both implementations already operate
near saturation at these values. The position-only result of 57.59% is close
to, but below, the approximately 59% result reported in [1]. This agreement
supports the reference study's conclusion that practical GPS position alone
does not fully characterize the optimal beam for a drone with three-dimensional
mobility.

### C. Effect of Height and Distance

**TABLE III**  
**TOP-1 IMPROVEMENT RELATIVE TO POSITION-ONLY PREDICTION**

| Added Modality | Top-1 Gain (percentage points) | Relative Gain (%) |
|---|---:|---:|
| Height | +12.12 | +21.04 |
| Height and Distance | +16.59 | +28.81 |

Adding height increases Top-1 accuracy from 57.59% to 69.71%, yielding a
12.12-percentage-point improvement. This result lies within the approximate
10--14-percentage-point gain for combined modalities reported in [1]. Adding
distance further increases Top-1 accuracy to 74.19%, producing a
16.59-percentage-point gain over position alone and a 4.48-percentage-point
gain over position-height.

![Top-1 test accuracy by sensing modality.](figures/top1_modality_comparison.png)

**Fig. 2.** Top-1 accuracy improvement as height, distance, and image sensing
are introduced.

The improvement demonstrates that height and distance provide complementary
geometric information that is not sufficiently represented by noisy
two-dimensional GPS coordinates. Distance particularly improves the model's
highest-confidence beam selection: compared with position-height, it improves
Top-1, Top-2, and Top-3 accuracy by 4.48, 0.97, and 0.88 percentage points,
respectively. Its Top-5 accuracy decreases slightly by 0.09 percentage points,
which is negligible at the present test-set size and should not be interpreted
as a statistically significant degradation.

### D. Vision-Aided Prediction

The refined ResNet-50 achieves 88.32% Top-1 accuracy, exceeding the strongest
position-family model by 14.14 percentage points and the position-only model by
30.73 percentage points. This result reinforces the principal finding of [1]:
an RGB image captures the drone's apparent location and orientation within the
base-station field of view more effectively than individual scalar sensing
measurements. The image model also achieves 99.56% Top-3 and 99.82% Top-5
accuracy. Consequently, restricting beam training to the model's five
highest-ranked candidates would retain the correct beam for nearly all tested
samples.

The present image model was refined from its best checkpoint using selective
unfreezing, frozen batch-normalization statistics, shuffled training samples,
and checkpoint averaging. This refinement increased test Top-1 accuracy from
87.62% to 88.32%, an absolute gain of 0.70 percentage points, while preserving
the previous checkpoint whenever validation performance did not improve.

![Initial and refined image model compared with the reference paper.](figures/image_model_refinement.png)

**Fig. 3.** Image-model refinement and direct comparison with the image
metrics reported in [1].

### E. Principal Findings

The comparative evaluation supports four findings.

1. Position-only prediction remains the weakest sensing-aided approach,
   confirming that two-dimensional GPS data alone is inadequate for robust
   drone beam selection.
2. Height and distance materially improve Top-1 accuracy, validating the
   reference study's conclusion that additional geometric sensing information
   is important for three-dimensional drone mobility.
3. Vision-aided prediction remains the strongest modality and exceeds the
   reference image Top-1 accuracy by 2.00 percentage points in the present
   implementation.
4. Top-3 and Top-5 image accuracies are close to 100% in both studies,
   indicating that sensing-aided prediction can substantially reduce exhaustive
   beam-training overhead.

### F. Limitations and Reporting Requirements

The observed accuracy differences should not be presented as evidence of
statistical superiority over [1] without additional controlled experiments.
The current implementation and [1] differ in beam-class definition, dataset
split, model depth, batch size, image initialization strategy, optimization
procedure, and treatment of missing sensor values. In addition, the reported
results are based on saved single-run outputs; repeated trials with fixed
splits and confidence intervals have not yet been reported.

For a strict reproduction study, the implementation should use the same
32-beam targets, 8,402/3,602 train-test split, two-hidden-layer position MLP,
batch size of 32, learning-rate schedules, and complete modality-aligned
samples specified in [1]. Until those conditions are satisfied, the defensible
claim is that the present findings reproduce the reference paper's qualitative
modality ranking and achieve strong quantitative results under a modified
experimental configuration.

## REFERENCE

[1] G. Charan, A. Hredzak, C. Stoddard, B. Berrey, M. Seth, H. Nunez, and
A. Alkhateeb, "Towards real-world 6G drone communication: Position and camera
aided beam prediction," in *Proc. IEEE Global Communications Conf.
(GLOBECOM)*, 2022, pp. 2951--2956, doi:
10.1109/GLOBECOM48099.2022.10000718.
