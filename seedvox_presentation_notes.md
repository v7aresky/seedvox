# SeedVox — Presentation Narratives

*Companion script for `seedvox_presentation.html` (16 slides, ~14 min). Each slide block gives you: the one-line takeaway the audience should leave with, the spoken narrative, the exact numbers to land, and the transition into the next slide. Read the *narrative* aloud (or adapt it to your voice); the *key numbers* are the non-negotiables to say once.*

**The over-arching story (say this to yourself before you start):**
> Speech engines today are a single bloated model guessing every detail token by token — so they hallucinate, they can't be steered, and emotion leaks away. SeedVox proves that *structure beats scale*: we separate **what** we say (explicit phonemes), **how** we say it (a JEPA prosody plan), and **who** says it (speaker conditioning) — and one renderer plays all three. On ~3 orders of magnitude less data than a big lab (~1/1000th of its budget), trained on a gaming GPU, the decoder measurably follows the plan. Results speak — then the audience listens.

---

## Act I — The case for structure

### Slide 1 · Title
**Time:** 0:20 · **Takeaway:** *SeedVox plans before it speaks — what, how, and who are split into clean components.*

**Narrative:**
"SeedVox is speech that plans before it speaks. Today's speech engines treat emotion like a brute-force problem: one giant autoregressive model guesses it token by token. But emotion is a **global state of mind** — it shapes the whole sentence before we even open our mouths. So we separate three questions that a single model shouldn't answer at once: the **what** — the words, planned explicitly as phonemes; the **how** — the prosody, planned by a JEPA world model that reads the whole sentence; and the **who** — the speaker identity, conditioned through two independent adapters. One acoustic decoder renders all three into audio tokens. That decomposition is the whole talk."

**Key numbers:** the four components — AR Phonetic Planner ("what") · JEPA ("how") · Speaker Conditioning ("who") · AR Decoder ("renderer"). Open-source, Apache 2.0.

**Transition:** "Why is brute force the wrong bet? Because of what it costs — and what you cannot do with it."

---

### Slide 2 · Motivation — against brute force
**Time:** 1:00 · **Takeaway:** *Scale hides the problems; structure solves them — from hallucination to pronunciation control.*

**Narrative:**
"Big labs win by scale — enormous transformers, enormous data, meaning regenerated token by token. We argue **structure beats scale** at the same budget. Scale buys quality the expensive way, but it hides three problems. First, **compute**: a monolithic AR model buries its expressiveness problems behind parameter counts most researchers can't afford. Second, **emotion**: treating emotion as an autoregressive roll is brute force — it drifts, and loses the whole-sentence tension that makes speech feel alive. Third, and this is the one we hear about most from users: **hallucination and no low-level control**. Frontier TTS hallucinates, stumbles on edge cases, and follows instructions inconsistently — and when it mispronounces a name, you cannot fix it. There is no handle on F0, no handle on timing. Those are exactly the problems SeedVox is designed to solve: an explicit AR phoneme planner gives pronunciation control and robust g2p; an explicit JEPA prosody latent gives you a handle on the F0 contour and timing; structural priors keep generation grounded and resist drift."

**Key numbers:** big compute · token-by-token drift · hallucination + no control.

**Transition:** "And the bet on structure extends to data: you don't need an internet scrape. Here's what we trained on."

---

### Slide 3 · Data — enough, not massive
**Time:** 1:15 · **Takeaway:** *276,529 utterances / 319.3 hours of curated, pre-cooked public data — expressiveness is measured, not captioned.*

**Narrative:**
"Structure beats scale extends all the way down to data. No proprietary corpora, no internet scrape — five public speech sets, **276,529 utterances, 319.3 hours** total, assembled like a recipe rather than scraped like raw material. HiFiTTS gives us a clean acoustic backbone at scale; GLOBES contributes breadth of speakers and reading styles; LibriTTS adds US audiobooks; LJSpeech is the expressiveness substrate with spontaneous-style reads; and DailyTalk provides real conversation give-and-take. The recipe philosophy: off-the-shelf corpora, curated roles, pre-cooked offline. Every utterance is resolved once into a `.pt` row — 16 RVQ tokens, text, phoneme ids, and F0/E/V aligned at the Mimi frame rate of 12.5 Hz — so training just streams cached rows, no live processing. A design detail that matters: we center prosody per utterance but keep the variance — that's what keeps the extremes available for the `exagg` dial later. The honest numbers: this is roughly **3 orders of magnitude smaller — about 1/1000th of a big lab's data budget**; counting the LibriVox-derived sets once — HiFiTTS and LibriTTS re-cut the same audiobooks — the unique corpus is actually **under 300 hours**. And it is **heavily audiobook-biased**: most of those hours are read-aloud narration — rich prosody *within* that style — with DailyTalk and LJSpeech providing only a conversational/spontaneous minority. Yet it's curated, so the prosody branch still learns from real F0/energy/voicing variation. One last point, because it defines our philosophy: typical TTS training data is increasingly captioned — 'lively', 'excited', 'annoyed'. We use **none** of that. Expression is measured from the audio directly — F0/E/V into a latent. The 'how' is measured, not captioned."

**Key numbers:** 116.0/125,844 · 104.6/81,299 · 53.8/33,113 · 23.0/12,500 · 21.9/23,773 → 319.3 h / 276,529. &lt;300 h unique (LibriVox sets deduped) · audiobook-biased. 12.5 Hz frame rate. ~3 orders of magnitude (~1/1000th) of a big-lab budget.

**Transition:** "With that data, what does the architecture actually look like?"

---

## Act II — The design

### Slide 4 · Architecture at a glance
**Time:** 1:00 · **Takeaway:** *Four dedicated components instead of one juggling network — what, how, who, and the renderer.*

**Narrative:**
"Instead of one giant network juggling everything, SeedVox splits the job into dedicated components. Text enters, and the path forks immediately. The 'what': the AR phonetic planner turns the text into explicit phonemes, then the linguistic-fusion block folds those phonemes back into the text via gated cross-attention — one unified representation the decoder can lock onto. The 'how', in parallel: the JEPA world model reads the whole sentence non-causally and emits a global prosody latent, injected at the decoder alongside — not fused into the text. The acoustic decoder then renders unified text plus prosody plan into audio tokens, and Mimi turns those tokens into a 24 kHz waveform.

The second diagram is Mimi itself — the reason that last step is so clean: audio goes in at 24 kHz, the neural encoder downsamples it to 12.5-Hz feature maps, a split residual quantizer separates a semantic stream from the residual codebook streams — the RVQ tokens the decoder predicts — and the neural decoder rebuilds the waveform from the sum. Two things to note on the main diagram. The voice — that dashed 'reference audio' — is optional, injected at the decoder: who speaks. And the orange box is training-only: a frozen prosody codec plus Mimi encoder that turn F0/E/V into a latent and supervise the planner. At generation time none of that exists — just text in, speech out."

**Key numbers:** 4 components · plan (B, 32, dim) · 24 kHz output.

**Transition:** "Now let's zoom into where the plan actually hits the decoder — twice."

---

### Slide 5 · Detailed pipeline — where the JEPA prosody hits the decoder
**Time:** 1:15 · **Takeaway:** *The plan conditions the decoder in two anatomical places — context cross-attention and the NAR depformer.*

**Narrative:**
"Here is the full pipeline. Text splits two ways: character ids through the text encoder, and BPE ids fused in through a gated add — the 'what' is being built from two granularities already. Above, the speaker encoder turns reference audio into a speaker vector. The text then feeds two planners: the AR phoneme predictor, and the JEPA prosody planner. Here's the subtlety worth your time — the prosody plan reaches the decoder **twice**, because our early probes showed the AR head's hidden state barely carries prosody. So first, the plan is FiLM-adapted by the speaker and sits as a block inside the cross-attention CONTEXT. Second, it is mean-pooled straight into the NAR depformer's adaptive layers, bypassing the prosody-lean path entirely. Everything downstream is conditional on the same context: the AR layer with speaker AdaLN, then the depformer predicting the residual 16 RVQ codebooks, then Mimi's decoder producing the waveform. The orange block is the frozen teacher — pyin gives F0/E/V, the prosody codec gives the ground-truth latent, and the cosine objective trains the planner. The speaker adaptation shown here in yellow is the 'who' we'll get to in a moment."

**Key numbers:** CONTEXT = [spk vec · speaker-FiLM'd prosody · unified text] · 16 RVQ codebooks · two conditioning sites.

**Transition:** "Before the deep dives, the principles — the design choices that make this work."

---

### Slide 6 · Key design choices
**Time:** 1:00 · **Takeaway:** *Global planning, learned latents, fused text, disentangled speaker, explicit control, the renderer, efficiency by construction.*

**Narrative:**
"Seven principles shape everything in SeedVox. **One:** prosody is a global state of mind, not a token-level dice roll — a learned latent shapes the whole utterance before generation. **Two:** latent, not features — F0/E/V only feed the frozen teacher; generation conditions on the learned latent. **Three:** unified linguistic fusion — a gated cross-attention layer folds the AR phoneme planning into the text backbone, so the decoder aligns to one grounded, fused representation instead of juggling parallel alignments. **Four:** dual-FiLM disentanglement — two speaker-modulated adapters keep articulation and rhythm/emotion separate, so conditioning never bleeds. **Five:** intervenable control — phonemes are explicit, so you can overwrite the phoneme string at runtime to fix a pronunciation. **Six:** the painter — the plans become audio through a streaming AR transformer handing off to a parallel NAR depformer, rendering the unified text plus prosody plan into discrete audio tokens the Mimi codec turns into speech. **Seven:** efficient by construction — gradient checkpointing, Fused AdamW, torch.compile; inference under 400 ms on an RTX 5090."

**Key numbers:** (B, 32, dim) plan · AR→NAR renderer → discrete audio tokens · <400 ms inference on one RTX 5090.

**Transition:** "Let's examine the 'what' — the linguistic structure."

---

### Slide 7 · The linguistic structure — the "what"
**Time:** 0:45 · **Takeaway:** *The 'what' is one fused representation — chars + BPE + explicit AR phonemes — aligned by attention, not a duration model.*

**Narrative:**
"The 'what' is not just phonemes — it's a fused, multi-level representation. Characters go through the text encoder; BPE adds word-level and morphology context through a gated add. Then the AR phoneme predictor produces explicit phoneme features — 128-symbol vocabulary — that are speaker-FiLM'd and folded into the text by the Linguistic Fusion layer through gated cross-attention. The result is one unified text representation, and the decoder aligns to it by a monotonic cross-attention layer. Two things make this more than a fancy encoder. First, the phoneme head is a regularizer *and* a control surface: pronunciation stays grounded even in fused text, and you can overwrite it. Second — and this is a point of pride — there is **no duration model**. Timing is driven by attention, which is how we avoid the stability collapses of duration-prediction systems."

**Key numbers:** vocab 128 · one unified representation · attention-based alignment, no duration model.

**Transition:** "Now the 'how' — the JEPA world model, the star of the design."

---

### Slide 8 · The JEPA World Model — the "how"
**Time:** 0:45 · **Takeaway:** *A JEPA predicts abstract latents, not signals — non-causal, sampleable, and measurable.*

**Narrative:**
"What is a JEPA? A Joint-Embedding Predictive Architecture predicts abstract latents rather than raw signals — it's supervised by a similarity or energy objective (a cosine metric in our case), never reconstruction. Why a JEPA instead of a pure AR LLM for prosody? An AR LLM must regenerate every token, so it guesses tone token by token and the global intent drifts. A JEPA reads the whole sentence first, non-causally; it spends no capacity on predictable acoustic detail; and it returns a plan you can sample and edit. One precision: the JEPA is the **architect, not the painter**. It decides *how* the sentence should sound; the renderer does the painting — the decoder turns plan and text into audio tokens, predicting all codebooks at once through the NAR path. The JEPA plans; the painter paints. It's supervised by the frozen prosody codec — F0/E/V into a latent — via that cosine/planning objective. It has learned stochasticity, so you can sample plans for controlled expressivity. And it's regularized with a small contrastive gap of 0.2 plus a learned-std term of 0.05 to fight mean-collapse — otherwise the planner would happily predict the average and produce a decorative latent. The result, against ground truth, is that a sampled plan is followed as faithfully as the true prosody: informative, not decorative."

**Key numbers:** non-causal · cosine objective · contrastive 0.2 · learned-std 0.05 · plan (B, 32, dim) · jepa_layers=2, heads=4.

**Transition:** "The 'who' — speaker identity and the renderer that brings it all together."

---

### Slide 9 · The "who" & the voice — speaker conditioning + acoustic decoder
**Time:** 0:45 · **Takeaway:** *Two FiLM adapters keep identity separated from expression; the decoder renders what + how + who.*

**Narrative:**
"The 'who' is the voice we hear — but it's not the decoder. The decoder is the renderer; identity is a conditioning signal. Two independent FiLM adapters turn the speaker latent into a per-frame scale and shift — a modulation. `film_phn` retunes the phoneme embeddings, controlling articulation and vocal tract; `film_prs` retunes the prosody tokens, controlling rhythm and intonation. Why two and not one flat speaker vector? Because one vector would bundle articulation together with prosody, and you'd get conditioning bleed: cloned voices would get the wrong rhythm, and a prosody edit would disturb articulation. Separate adapters each learn a clean job — precise cloning, no bleed. Then the renderer: a streaming AR transformer over 16 codebooks, cross-attending to the fused text with speaker AdaLN per layer, followed by a NAR depformer that predicts the residual codebooks with speaker and prosody AdaLN. The Mimi codec encodes the waveform into the token stream the decoder reconstructs."

**Key numbers:** film_phn / film_prs · n_q=16 · dim=512, layers=6, depformer=10, heads=8, card=2048.

**Transition:** "Now the contract that makes all of this accountable — every loss has a job, and every job has a test."

---

## Act III — The evidence

### Slide 10 · Losses & tests — every term has a job
**Time:** 1:15 · **Takeaway:** *A four-term loss keeps the who/what/how split honest — and each is verified by a targeted test.*

**Narrative:**
"Four loss terms, each owning one responsibility. `loss_ar` — audio quality — cross-entropy over the RVQ codebooks, with a curriculum from 4 to 16 codebooks over the first 50k steps. `loss_ph` — the what — cross-entropy on the phoneme head, keeping pronunciation grounded even inside fused text. `loss_jepa` ×2.0 — the how — and it's really three terms in one: cosine similarity between the plan and the true latent so the plan points *like* the prosody — shape, not volume; a small contrastive gap of 0.2 so the plan must beat every other text's latent in the batch — that's what stops mean-collapse; and a learned-std term of 0.05 that keeps the sampler's scale near its init — expressive variety at generation without explosion. And `loss_cycle` — plan-follow — the cosine between the *generated* audio's prosody and the plan it was conditioned on; this is what forces the decoder to actually use the latent instead of reading text. A duration auxiliary exists but is deliberately unused — alignment is attention-based. And because every term claims a behavior, every claim has a repo test: a smoke train on the real config, Mimi fidelity on partial codebooks, a pitch sweep that must actually move predicted pitch, g2p/phoneme correctness, and a plan-follow trio scored every eval. Finally, the intelligibility guard: the demo sample scores about 6% WER on Whisper — 14 out of 222 words — and every edit is the recognizer misreading out-of-vocabulary names like 'SeedVox' as 'seedbox', so the spoken words themselves are intact."

**Key numbers:** weights 1.0 / 0.5 / 2.0 / 0.5 · curriculum 4→16 over 50k steps · contrastive 0.2 · std 0.05 · duration aux 0.08 unused · ~6% WER (14/222).

**Transition:** "Now the results — the measurement that matters most."

---

### Slide 11 · Results — the plan is followed, not ignored
**Time:** 1:15 · **Takeaway:** *gen_ar == ar — sampling the plan costs nothing; the decoder genuinely follows it.*

**Narrative:**
"Across 200 evaluation batches — 2,000 sentences — one reference speaker, generated on a single RTX 5090. Here's how to read the table. The first two rows are audio-token cross-entropy: lower is better, and below 1.5 is quite good for a multilingual codec. The last two are cosine distances — 0 is identical, 1 is unrelated — and each compares a different pair: `plan_cos` puts the sampled plan against the true prosody, `follow_cos` puts the rendered audio's prosody back against the plan. With ground-truth prosody the decoder reaches 1.070. With a JEPA-*sampled* plan — no oracle, just the model's own prediction — it reaches **1.070**. Identical. The sampled plan lands 0.237 from the ground-truth prosody (`plan_cos`), and the decoder follows it with a distance of 0.317 (`follow_cos`) — close enough that quality is unchanged. Here's the argument for why this is the key result: if the decoder were ignoring the plan and just reading text — the shortcut collapse — gen_ar would be measurably worse. It isn't. The decoder follows the plan, the latent is genuinely informative."

**Key numbers:** 2,000 sentences · ar 1.070 · gen_ar 1.070 · plan_cos 0.237 · follow_cos 0.317 (both cosine distance).

**Transition:** "Getting there was not clean — and the failures are the most honest part of the story."

---

### Slide 12 · Lessons learned — plugging JEPA latents into an AR decoder
**Time:** 1:15 · **Takeaway:** *The hardest integration was getting a global latent into a causal decoder — three failure modes, five fixes.*

**Narrative:**
"Feeding a global, non-causal prosody latent into a sequential, causal token decoder was the hardest integration in the project — and it failed in three instructive ways. **First, text-only shortcut collapse:** when the decoder is conditioned on the *predicted* plan — a deterministic function of text — the latent becomes redundant and the decoder quietly ignores it. **Second, a prosody-lean AR head:** probing the decoder's hidden state showed it barely carries prosody, so passing the latent only through AR context leaked nothing downstream. **Third, planner mean-collapse:** without extra pressure the predicted plans hug the average direction — globally informative but not discriminative across texts. Each failure pointed to a fix. Mix teacher-forcing: condition on ground-truth prosody half the time so the decoder learns latent-to-audio, and sample a plan the rest of the time to keep inference in distribution. Contrastive plus std regularization pushed the planner away from collapse. Injecting mean-pooled prosody directly into the NAR codebook predictor bypassed the prosody-lean path. The exagg dial interpolates between the null and sampled plans at inference — expressiveness control with no retraining. And the cycle loss made plan-following measurable. Every number you saw is the result of this journey."

**Key numbers:** 3 failure modes → 5 fixes (mix teacher-forcing · contrastive+std · direct depformer AdaLN · exagg dial · cycle loss).

**Transition:** "None of this would matter if it weren't frugal. It is — by construction."

---

## Act IV — The payoff

### Slide 13 · Frugal & controllable by design
**Time:** 0:45 · **Takeaway:** *284M parameters, one consumer GPU, local and offline — with inference-time controls instead of retraining.*

**Narrative:**
"Because we split the job and baked structure in, the whole system is small and controllable. A seeded run is reproducible. You can walk the latent space along the prosody or speaker axis. Voice cloning is a LoRA adapter, not a retrain. CFG tunes reference fidelity: it contrasts a full-context decode against a text-blind, prosody-grounded one, so you dial how closely the audio follows the reference — not retrained. There's a temperature for each stage. And the whole thing runs with gradient checkpointing, Fused AdamW and torch.compile at under 400 ms inference on a single RTX 5090, fully local and offline. The hardware democratization point is the thesis: SOTA-quality speech on a gaming laptop makes frontier voice technology approachable, sovereign — your data, your process, your output — and hackable, for students and researchers."

**Key numbers:** RTX 5090 · <400 ms · LoRA / CFG / exagg / per-stage temperature · fully local.

**Transition:** "Enough claims — let's listen."

---

### Slide 14 · Listen
**Time:** 0:45 (plus playback) · **Takeaway:** *No cherry-picking — one speaker, one demo, all prosody sampled by the JEPA.*

**Narrative:**
"This is a ten-sentence demo generated on a single RTX 5090. The reference voice and the training data are LJ Speech — an audiobook corpus: read speech, monotone, low expressiveness, no conversational dynamics. That's the honest baseline. All the prosody you hear is sampled from the JEPA world model — no manual tuning — which is exactly the claim of this talk: what you say is planned, how you say it is planned, who says it is conditioned, and the renderer follows. Note how much character the 'how' carries even from flat audiobook data — expressive conversational reference data will sound considerably more dynamic."

**Key numbers:** 10-sentence demo · docs/seedvox_demo.mp3 · LJ Speech reference · JEPA-sampled prosody.

**Transition:** "Where does this go next?"

---

### Slide 15 · Roadmap
**Time:** 0:30 · **Takeaway:** *Done now; next is data and finetuning; then the streaming future.*

**Narrative:**
"Where we stand: the hybrid AR-JEPA baseline is done — an autoregressive discrete-transformer decoder plus JEPA, frozen prosody teacher, dual-FiLM conditioning, and plan-following verified. Next is data: expressive conversational corpora with longer-form context and natural-language emotion and style tags embedded in the text — the supervision our style head and planner want, and the direction the field is heading. Beyond data, two explorations: a continuous-latent acoustic decoder so the prosody signal lands without a quantization bottleneck; and one honest admission — mid- and post-training preference finetuning, DPO or GRPO, has not been explored yet, because we've worked with a very limited data budget; what we *do* ship today is LoRA adaptation for fast voice changes. And beyond everything: streamability — chunk-wise inference and a full-duplex spoken channel, so a system can listen and speak at the same time, the shape of real conversation."

**Key numbers:** done = plan-follow verified · next = conversational+tagged data · beyond = continuous-latent decoder, DPO/GRPO unexplored, full-duplex streaming · today = LoRA.

**Transition:** "Close on what matters: what/how/who, planned before it's spoken."

---

### Slide 16 · Closing / Takeaways
**Time:** 0:30 · **Takeaway:** *Structure beats scale: what, how, who planned — 284M params, <400 ms, Apache 2.0, fully open-sourced (training, inference, configs) — and people are listening.*

**Narrative:**
"The takeaway is the title: what you say, how you say it, who says it — planned before it is spoken. SeedVox is smarter speech AI: structural priors instead of brute-force scale. The 'what' is explicit AR phonemes fused with BPE and chars into one representation. The 'how' is a JEPA plan, and we measured that the decoder follows it — gen_ar equals ar, no shortcut collapse. The 'who' is conditioned, not guessed. And it's fully open-sourced: training, inference, and model configurations are all in the repo — 284 million parameters including the renderer, under 400 milliseconds on a single consumer GPU, Apache 2.0. And it learns fast — after just 12 epochs on the single-speaker LJ Speech corpus, trained in under an hour on one GPU, it already speaks intelligibly and follows the input text. That's an iteration cycle researchers can hold in their hands: hypothesize, retrain, listen, learn — in an afternoon. And — thank you to everyone who has already found it — it's getting attention: over 25,000 views on LinkedIn, and PhD students reaching out to collaborate. That traction is the validation that frugality and structure are what the community wants. The code, the checkpoints pipeline, and the demo are online — listen, build, and reach out."

**Key numbers:** what/how/who · 284M params · <400 ms · 12 epochs on LJ Speech, <1 h on one GPU · Apache 2.0 · +25k LinkedIn views · PhD collaborators.

**Closing URLs:** github.com/v7aresky/seedvox · https://v7aresky.github.io/seedvox/

---

## Delivery checklist
- **Act I (1–3):** the wager. Land the *measured, not captioned* idea early — it recurs.
- **Act II (4–9):** the design. The pipeline slide (5) is the one to point at a lot; everything else reinforces it.
- **Act III (10–12):** the evidence. Say `gen_ar == ar` out loud twice — it's the single most quotable result.
- **Act IV (13–16):** the payoff. Keep slide 15 fast; end on the traction and the URL.
- Keep the *numbers* as spoken anchors — the narrative around them can flex to the audience and time.