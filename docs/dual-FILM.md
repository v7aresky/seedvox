# Dual-FiLM Speaker Conditioning

## Overview
To improve speaker-specific synthesis quality and disentangle phonetic and prosodic features from speaker identity, we have implemented a dual-FiLM (Feature-wise Linear Modulation) architecture. 

## Implementation
The implementation introduces two independent `FiLMAdapter` modules within `ExplicitPlannerModel`:
- **`self.film_phn`**: Modulates phoneme embeddings based on speaker latents.
- **`self.film_prs`**: Modulates prosody tokens based on speaker latents.

These adapters compute scale ($\gamma$) and shift ($\beta$) parameters derived from the speaker latents, which are applied to the respective embeddings as:
$$\text{Modulated} = (1 + \gamma) \cdot \text{Embedding} + \beta$$

## Conclusion
The dual-FiLM architecture provides a more robust and flexible mechanism for speaker conditioning. By allowing independent, speaker-specific transformations for phonetics and prosody, the model can better learn how speaker identity affects both the vocal tract configuration (phonetics) and rhythmic/intonational patterns (prosody). This enhancement is expected to improve synthesis quality and increase speaker similarity, though it requires retraining due to the added learnable parameters.
