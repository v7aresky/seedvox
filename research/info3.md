🚀 Announcing SeedVox v3: The "Anchored" Cross-Attention Pivot

  We are proud to introduce SeedVox v3, a major architectural evolution in our neural audio codec research. While many contemporary systems utilize the
  Prefix-LM (unified self-attention) paradigm, v3 represents a strategic pivot to an Encoder-Decoder architecture with dedicated Cross-Attention,
  specifically optimized for high-fidelity performance at small-to-medium scales.

  🏗️ The Breakthrough: Positional Anchoring
  Traditional Prefix-LM systems often suffer from alignment instability at smaller parameter counts. Because text and audio share a single sequence,
  changing the input length shifts the absolute "address" of the text, forcing the model to constantly recalculate its focus.

  SeedVox v3 solves this by decoupling the positional spaces. Both the text and audio sequences are anchored at Position 0 in their respective
  transformers. The decoder no longer has to "search" for the text context; it has a dedicated, constant port to the text blueprint at every layer via
  Cross-Attention.

  🎙️ Speaker Perceiver Encoder: Timbre Stability
  In place of sequential prompting, v3 introduces the Speaker Perceiver encoder. This specialized module extracts a fixed set of global style latents
  from reference audio.
   * How it works: These latents are injected into the Cross-Attention context memory and queried by the decoder at every single layer.
   * The Benefit: This ensures absolute timbre stability. The model cannot "forget" the voice identity during long utterances. The speaker’s unique
     vocal character is "clamped" to the generation process, leading to superior zero-shot voice cloning.

  🧠 Joint Character-BPE Intelligence
  SeedVox v3 maintains a streamlined end-to-end pipeline by using a Joint Character and BPE Residual Encoder. This approach provides the model with a
  multi-resolution understanding of the input without the need for external phonetic tools:
   * Character-Level Resolution: Characters provide the high-resolution sequence required for the model to "read" exactly what is on the page.
   * BPE Semantic Context: Word-level BPE embeddings are added as residuals to the character features. This provides the deep semantic context needed to
     resolve homograph ambiguities (e.g., "read" vs. "read" based on sentence structure) and ensures more natural, human-like phrasing.

  💎 Why SeedVox v3 Outperforms Other Systems
   1. Instant Alignment: By using explicit Cross-Attention, v3 achieves stable text-following almost immediately, bypassing the long "babbling" phase of
      unified models.
   2. Style-Content Separation: Decoupling the Speaker Perceiver encoder from the text sequence prevents the speaker's style from being incorrectly tied
      to specific words.
   3. Hierarchical Curriculum Learning: Leveraging the Mimi codec’s hierarchy, v3 trains on the core "skeleton" of speech ($n_q=1$) first, mastering
      rhythm and timing before focusing on acoustic texture.
   4. Hardware Democratization: While massive unified models require data-center GPUs, v3 is the most stable architecture for small-to-medium deployment
      (50M–500M parameters), delivering industry-grade quality on consumer hardware.

  🛠️ Technical Stack
   * Codec: Mimi (RVQ-based Neural Audio Codec)
   * Backbone: StreamingTransformer with KVCache Optimization
   * Style Extraction: Speaker Perceiver Encoder via Cross-Attention
   * Text Processing: Joint Character + BPE Residual Encoder

  SeedVox v3 is a re-imagining of how a compact Transformer should "understand" text and "generate" speech.

  Training is active. The anchor is set.
