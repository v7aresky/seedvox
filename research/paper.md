# SeedVox: Explicit Phoneme and Prosody Planning with Speaker-Conditioned Dual-FiLM Modulation

## Abstract
Recent advancements in neural text-to-speech (TTS) have significantly improved naturalness, yet explicit control over phonetics and prosody—particularly under varying speaker conditions—remains a challenge. Furthermore, state-of-the-art systems often rely on "brute-force" approaches, utilizing massive model architectures, enormous datasets, and compute-intensive training farms, resulting in inefficient convergence and restricted research access. We introduce **SeedVox**, an elegant, science-inspired speech synthesis system that leverages the functional distinction between sequential and global acoustic generation. SeedVox features an **Explicit Phonetic Planner** for autoregressive linguistic sequencing and a **JEPA-based World Model** for global prosodic and emotional state modeling. To address speaker-dependent realization, we propose a novel **Dual-FiLM (Feature-wise Linear Modulation)** architecture, enabling independent, speaker-conditioned transformations of phoneme and prosody embeddings. SeedVox is designed to be lean, achieving rapid training convergence on consumer-grade hardware, thereby democratizing high-fidelity speech synthesis research.

## 1. Introduction
While modern neural TTS models achieve high naturalness, they often suffer from the limitations of implicit conditioning, conflating speaker identity with prosodic and phonetic realization. Moreover, the prevailing paradigm in industry relies on large-scale brute-force techniques: training monolithic transformers on massive datasets using extensive compute clusters. These models frequently require exorbitant training durations before they learn to synthesize intelligible speech, rendering them inaccessible to academic researchers and prohibitive for educational purposes.

**SeedVox** proposes an alternative paradigm: a frugal, intelligent, and science-inspired approach to neural audio synthesis. We argue that not all components of speech synthesis share the same structural requirements. Text, phonetic tokens, and acoustic tokens are inherently autoregressive (AR) processes—they are produced incrementally, sound by sound, word by word. Conversely, emotion and prosody represent a more "global" state of mind, dictating the overall expressive contour of an utterance rather than just its immediate sequence. By incorporating these explicit structural priors into the architectural design, SeedVox achieves rapid learning and high controllability with a fraction of the data and compute required by conventional brute-force methods.

Our primary technical contributions are:
- **Sequential AR Planning:** Dedicated AR transformers that enforce structural stability in phonetic and acoustic sequencing, reducing hallucination via End-of-Sequence (`<EOS>`) prioritization.
- **Global Prosodic World Modeling:** A JEPA-based world model that treats prosody and emotion as global states, capturing high-level expressive intent effectively.
- **Speaker-Conditioned Dual-FiLM:** A mechanism that applies dynamic, feature-specific modulation to phoneme and prosody embeddings based on speaker identity. This moves beyond simple concatenation, allowing the model to learn speaker-dependent warping of phonetic and prosodic manifolds.
- **Rapid Convergence & Frugality:** Through architectural intelligence, SeedVox demonstrates significantly faster learning, making high-quality synthesis training viable on single-GPU personal workstations.

## 2. The SeedVox Architecture
The SeedVox architecture centers on the `ExplicitPlannerModel`, which harmonizes AR sequential modeling for phonetics with a JEPA-based world model for prosodic expressivity.

### 2.1 Sequential vs. Global Planning
SeedVox treats speech generation as a tiered task:
- **AR Planning (The "Thinker"):** The `PhoneticPlanner` operates on joint character and BPE features to predict an explicit phoneme sequence. By utilizing a weighted cross-entropy loss that prioritizes the End-of-Sequence (`<EOS>`) token, we significantly reduce phoneme hallucinations, ensuring the autoregressive planner terminates reliably.
- **JEPA World Model:** Recognizing that prosody is a global state, we utilize a Joint-Embedding Predictive Architecture (JEPA) block to model prosody and emotional expression at a higher level, capturing the holistic "state of mind" of the speaker.

### 2.2 Speaker-Conditioned Modulation via Dual-FiLM
To achieve nuanced speaker control, we replace traditional embedding concatenation with a **Dual-FiLM** approach. Two independent `FiLMAdapter` modules map speaker latents to affine transformation parameters ($\gamma, \beta$) for phoneme and prosody embeddings:
$$\text{Modulated} = (1 + \gamma(s)) \cdot \text{Emb} + \beta(s)$$
This design allows the model to learn distinct speaker-specific transformations for vocal tract configuration (phonetics) and intonation/rhythm (prosody), fulfilling the dual-objective of disentanglement and expressivity.

## 3. Training and Stability
Given the complexities of parallel data generation, we implemented a robust training collation pipeline. This pipeline includes resilient generation through multi-backend fallback and consistency monitoring via a cumulative failure-rate threshold, enabling fail-fast behavior when the environment cannot support high-throughput data generation.

## 4. Conclusion
SeedVox offers a sophisticated, yet accessible approach to controllable speech synthesis. By explicitly modeling phonemes and prosody within a speaker-aware framework and optimizing for resource-constrained hardware, SeedVox provides a lean and transparent platform for researchers. Our work demonstrates that architectural elegance can effectively substitute for excessive computational brute force, empowering a broader community to participate in the cutting edge of neural audio innovation.
