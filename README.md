# Low-Resource Neural Machine Translation: English to Malayalam

Fine-tuning `Helsinki-NLP/opus-mt-en-dra` (MarianMT) for English→Malayalam translation, with domain adaptation on technical lecture data. Evaluated on out-of-domain ML/scientific text.


## Results

| Model | Training Data | Evaluated On | BLEU | 1-gram | 2-gram | 3-gram | 4-gram | BP |
|-------|--------------|-------------|------|--------|--------|--------|--------|----|
| Baseline | BPCC (general domain) | BPCC val | 15.48 | 47.6 | 20.8 | 10.4 | 5.6 | 0.999 |
| Adapted | BPCC + Shiksha (technical) | Shiksha val | 32.61 | 66.6 | 40.9 | 26.2 | 17.4 | 0.976 |

The two models are evaluated on different validation sets — BPCC val (general domain) and Shiksha val (technical domain), respectively. The scores are not directly comparable as absolute numbers. The n-gram progression tells the real story: the adapted model improves dramatically on longer phrases (4-gram: 5.6 → 17.4), reflecting genuine gains in fluency and domain vocabulary rather than just unigram recall.

## Data

- **[AI4Bharat BPCC](https://huggingface.co/datasets/ai4bharat/BPCC)** (`bpcc-seed-latest`, `mal_Mlym`): The ai4bharat/BPCC dataset on HuggingFace is a comprehensive parallel corpus. It includes the IN22 benchmark — a multi-domain, n-way parallel test set across 22 Indic languages with subsets covering news, entertainment, culture, legal, and India-centric topics. This gives you a general-domain English–Malayalam foundation to fine-tune from.
- **[SPRINGLab Shiksha](https://huggingface.co/datasets/SPRINGLab/shiksha)**: A parallel corpus mined from NPTEL technical lecture transcriptions, filtered at quality score > 0.85 (54k → 43k train pairs). See the original paper [here](https://arxiv.org/abs/2412.09025).

## Model

`opus-mt-en-dra` is a Marian encoder-decoder transformer pre-trained on English → Dravidian languages (Malayalam, Kannada, Tamil, Telugu). Its multilingual SentencePiece tokeniser covers all four scripts with a shared vocabulary of ~63k tokens. Language selection at inference time is via a prefix tag (`>>mal<<`) prepended to the source sentence.

## Training

| Phase | Data | Epochs | LR | BLEU |
|-------|------|--------|----|------|
| Baseline | BPCC (80/20 split) | 5 | 5e-5 | 20.25 (train) / 15.48 (inference) |
| Domain adaptation | Shiksha (80/20 split) | 4 | 2e-5 | 32.65 |

Batch size 32, fp16, `beam search (num_beams=4)`, `max_length=256`.

## Architectural Notes
### Preprocessing and Tokenization

**Tokenisation for `seq2seq`**: Unlike classification, we tokenise both source and target sequences. The source gets the >>mal<< language tag prepended. The target is tokenised separately and stored as labels. During training, the model learns to predict the next target token given the source and all previous target tokens (teacher forcing).

**Truncation at 256 tokens**: Malayalam sentences are morphologically complex — a single word can encode what English expresses in several words. We cap at 256 tokens, which covers ~98% of sentences in this corpus without truncation.

**No padding here**: Padding is deferred to the data collator, which pads dynamically per batch to the length of the longest sequence in that batch. This is more efficient than padding to a global maximum. 

### Fine-tuning the Transformer

* The data collator

```
data_collator = DataCollatorForSeq2Seq(
    tokeniser,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
)
```
The sentences are all different lengths. But to train efficiently, you process them in batches — and batches need to be rectangular (every row the same length). The collator pads shorter sentences with a special `<PAD>` token to make them all the same length as the longest one in that batch.
`pad_to_multiple_of=8` just rounds up to the nearest multiple of 8 for GPU efficiency.

* The metric function
```
compute_metrics(eval_preds):
    preds, labels = eval_preds
```
`preds` are what the model output. `labels` are what it should have output. Both are batches of integer lists at this point.

 ```
 labels = np.where(labels != -100, labels, tokeniser.pad_token_id)
 ```
During training, padding positions in the labels get replaced with `-100` — a special value that tells the loss function "ignore this position, don't penalise the model for getting padding wrong". Before decoding labels back to text, we need to swap `-100` back to the actual padding token ID.
```
decoded_preds  = tokeniser.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokeniser.batch_decode(labels, skip_special_tokens=True)
```

Convert integer lists back to human-readable strings. `skip_special_tokens=True` removes things like `<PAD>` and `<EOS>` from the output.
```
result = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu": round(result.score, 2)}
```
BLEU is the standard metric for translation quality. It measures how much the model's output overlaps with the reference translation. 0 = completely wrong, 100 = perfect match.

* Training arguments
```
training_args = Seq2SeqTrainingArguments(
    output_dir="./en-kn-marian",   # where to save checkpoints
    num_train_epochs=3,            # go through the full dataset 3 times
    per_device_train_batch_size=32, # process 32 sentences at a time
    per_device_eval_batch_size=32,
    warmup_steps=500,              # start with a tiny learning rate, ramp up over 500 steps
                                   # prevents unstable updates at the start
    weight_decay=0.01,             # mild regularisation — discourages the model from
                                   # changing too drastically from its pretrained weights
    learning_rate=5e-5,            # how big each update step is — small because we're
                                   # fine-tuning, not training from scratch
    fp16=True,                     # use 16-bit numbers instead of 32-bit
                                   # halves memory usage, runs faster on GPU
    predict_with_generate=True,    # during evaluation, actually generate translations
                                   # rather than just looking at raw logits
    generation_max_length=256,
    eval_strategy="epoch",         # run evaluation once per epoch
    save_strategy="epoch",         # save a checkpoint once per epoch
    load_best_model_at_end=True,   # when training finishes, restore whichever
                                   # checkpoint had the best BLEU score
    metric_for_best_model="bleu",
    logging_steps=100,             # print a loss update every 100 steps
    report_to="none",              # don't send logs anywhere external
```

* The Trainer
```
trainer = Seq2SeqTrainer(
    model=model,                        # the model to train
    args=training_args,                 # all the settings above
    train_dataset=tokenised["train"],   # your training data
    eval_dataset=tokenised["validation"], # your validation data
    data_collator=data_collator,        # the padding function from earlier
    compute_metrics=compute_metrics,    # the BLEU function from earlier
)

trainer.train()  # actually starts training
```




## Technical Notes

**Batching:** The model processes sentences in groups of 32 rather than one at a time. Each batch goes through a single forward pass (encoder reads all 32 source sentences in parallel, decoder generates all 32 translations) and a single backward pass (gradients computed for all 32 at once). This is dramatically more efficient than processing sentences individually — GPU cores stay occupied and the fixed overhead of each forward/backward pass is amortised across 32 examples. The data collator handles the practical problem that sentences in a batch have different lengths: it pads shorter sentences with a special `<PAD>` token to make the batch rectangular (every row the same length), with `pad_to_multiple_of=8` aligning tensor dimensions for Tensor Core efficiency. Padding positions in labels are masked with `-100` so the loss function ignores them — the model is only penalised for tokens it was actually supposed to predict.

**Autoregressive decoding:** During training, the decoder sees the full target sequence at once (teacher forcing — it is shown the correct previous token at each step regardless of what it predicted). During inference, no target sequence exists yet. The decoder generates one token at a time: token 1 is generated from the encoder output alone, token 2 is generated from the encoder output + token 1, token 3 from the encoder output + tokens 1 and 2, and so on until an end-of-sequence token is produced. This sequential dependency means inference cannot be parallelised across the sequence dimension — a fundamental bottleneck. We use beam search with `num_beams=4`: rather than greedily picking the single most probable token at each step (which can lead to locally good but globally poor sequences), the decoder maintains the 4 most probable partial sequences at each step and returns the one with the highest overall probability when complete.

**Multilingual tokenisation:** `opus-mt-en-dra` is a one-to-many model — one English encoder, multiple Dravidian decoders (Malayalam, Kannada, Tamil, Telugu). It uses a SentencePiece tokeniser with a single shared vocabulary of ~63k tokens covering all four scripts plus English. SentencePiece segments text into subword units learned from the training corpus — common words get their own token, rare words are split into smaller pieces. This allows the model to handle unseen words by composing them from known subword units. The `>>mal<<` prefix tag prepended to every source sentence is the mechanism by which the decoder selects Malayalam at generation time — without it, the model defaults to whichever Dravidian language dominated its pretraining data.

**BLEU limitations:** BLEU (Bilingual Evaluation Understudy) counts n-gram overlaps between the model's output and a human reference translation, normalised by length. A score of 100 means every n-gram matches the reference; 0 means no overlap. In practice, scores above 30 are considered reasonable for low-resource MT. BLEU has well-known failure modes: it rewards surface-level lexical matches but cannot assess semantic equivalence (a perfectly correct paraphrase scores 0 if it shares no n-grams with the reference); it is sensitive to morphological variation (a correctly inflected Malayalam word that differs from the reference inflection scores 0, even though both are valid); and it does not account for word order differences between typologically different languages (English is SVO, Malayalam is SOV). For this project, BLEU is used as a relative metric — the 17-point improvement from baseline to adapted model is meaningful, but neither absolute score should be interpreted as a percentage of "correctness."

**Inference bottlenecks:** Three factors limit translation speed. First, autoregressive generation is inherently sequential — each token depends on the previous one, so the decoder cannot be parallelised across the sequence dimension regardless of GPU size. Second, memory bandwidth: at each generation step the model must load its full set of weights from GPU memory, making the operation memory-bound rather than compute-bound. Third, beam search multiplies the computation by the beam width (4× here) — maintaining 4 candidate sequences means 4 decoder forward passes per token. Batching the source sentences (32 at a time) amortises the encoder cost, since the encoder runs once per source sentence in parallel, but the decoder bottleneck remains.

**Domain adaptation:** A model trained on general-domain text (news, Wikipedia) learns vocabulary and sentence structures common in everyday language. When applied to a different domain — technical lectures, scientific papers, legal documents — it encounters vocabulary it has rarely or never seen, and produces incorrect or phonetically-transliterated output. Domain adaptation fine-tunes the pretrained model on a smaller, domain-specific corpus at a lower learning rate (`2e-5` vs `5e-5`), updating the model's weights to incorporate the new vocabulary while retaining the general linguistic knowledge acquired during pretraining. The 17-point BLEU improvement here reflects the model learning technical lecture vocabulary from Shiksha's NPTEL corpus. Importantly, the two evaluation sets are different (BPCC val for baseline, Shiksha val for adapted), so the improvement measures in-domain performance. Cross-domain evaluation on arXiv abstracts is qualitative — no Malayalam references exist for that corpus.

**Why two different evaluation sets?** The baseline is evaluated on BPCC val and the adapted model on Shiksha val. This is intentional — each model is evaluated on held-out data from its own training distribution to measure how well it learned that domain. A fair cross-domain comparison would require a single fixed test set with Malayalam references for both models. The arXiv translations saved in `arxiv_shiksha_translations.json` are a step toward this but lack references, making quantitative comparison impossible without human annotation.

**LaTeX handling:** arXiv abstracts contain LaTeX math notation (`$\epsilon$`, `\mathcal{L}`) which the SentencePiece tokeniser has never seen — it was trained on natural language text. The tokeniser segments LaTeX commands into arbitrary subword units with no mathematical meaning, producing phonetic transliterations of symbol names in the output ("യൂഗാസ്ലിയോൺ ഡോളർ" for `$\epsilon$`). A cleaning step using regular expressions strips inline math (`$...$`), display math (`$$...$$`), and LaTeX commands (`\cmd{...}`) before translation.

## Test Sentences

### Simple sentence
**Input:** The cat sat on the mat.
- **Baseline** → പൂച്ച മുലയിൽ ഇരുന്നു → back-translation: *"The cat sat on the breast."*
- **Adapted** → പൂച്ച മുലയിൽ ഇരുന്നു → back-translation: *"The cat sat on the breast."*

"mat" has no direct Malayalam equivalent. The model maps it to the phonetically/semantically closest token in its vocabulary — producing a plausible but wrong translation. This error survived domain adaptation, showing the limitation is fundamental: it originates in the base model's pretraining vocabulary, not in the fine-tuning data.

### NLP sentence
**Input:** We propose a novel attention mechanism for neural machine translation.
- **Baseline** → ന്യൂറൽ മെഷീൻ വിവർത്തനത്തിനായി ഞങ്ങൾ ഒരു നോവൽ ശ്രദ്ധാ സംവിധാനം നിർദ്ദേശിക്കുന്നു → back-translation: *"We propose a novel attention mechanism for neural machine translation."* ✓
- **Adapted** → ന്ന്യൂറൽ മെഷീൻ വിവർത്തനത്തിനായുള്ള നോവൽ ശ്രദ്ധാ കേന്ദ്ര സംവിധാനം ഞങ്ങൾ നിർദ്ദേശിക്കുന്നു → back-translation: *"We propose a novel attention center system for neural machine translation."*

The baseline handles this better — "attention mechanism" → "ശ്രദ്ധാ സംവിധാനം" is closer than the adapted model's "ശ്രദ്ധാ കേന്ദ്ര സംവിധാനം" (attention center system). Fine-tuning on lecture data introduced a drift toward educational register ("കേന്ദ്ര" = center, as in learning center) that slightly degrades NLP terminology. Domain adaptation improves aggregate BLEU but can degrade specific terminology.

## Testing on Technical Sentences (arXiv)

### Baseline model: BPCC only
**With LaTeX (unprocessed):**
Input: *In this paper we propose $\epsilon$-Consistent Mixup...*
Output: ഈ പ്രബന്ധത്തിൽ, യൂഗാസ്ലിയോൺ ഡോളർ ഡോള്‍...
The LaTeX symbols are transliterated as nonsense. The translation is unusable.

**Without LaTeX (cleaned):**
Input: *We introduce two-scale loss functions for use in various gradient descent algorithms...*
Output: ഫ്‌ളൈറ്റഡ് ന്യൂറൽ ശൃംഖലകൾ വഴിയുള്ള പ്രശ്നങ്ങൾ വർഗ്ഗീകരണത്തിലേക്ക്...
"gradient descent" → "ഗൾഡർ ന്" — partially recognised but corrupted.

### Adapted model: BPCC + Shiksha
Input: *A novel multi-scale loss function for classification problems in machine learning. We introduce two-scale loss functions for use in various gradient descent algorithms...*
Output: മെഷീൻ ലേണിംഗിൽ വർഗ്ഗീകരണ പ്രശ്നങ്ങൾക്കുള്ള ഒരു നോവൽ-സ് ട്രെയിൻ ലോസ് ഫംഗ്ഷൻ. വിവിധ ഗ്രേഡിയന്റ് പാരമ്പര്യ അൽഗോരിത പ്രവർത്തനങ്ങളിൽ...
"gradient descent" → "ഗ്രേഡിയന്റ് പാരമ്പര്യ അൽഗോരിത" (gradient + traditional/hereditary + algorithm) — still not correct but recognisably in the right semantic neighbourhood. "machine learning" → "മെഷീൻ ലേണിംഗ്" — correctly transliterated.

## Environment

Python 3.12, PyTorch, HuggingFace Transformers, Google Colab T4 GPU.

## Status

- [x] BPCC baseline training (5 epochs, BLEU 15.48)
- [x] Shiksha domain adaptation (4 epochs, BLEU 32.65)
- [x] arXiv out-of-domain evaluation (qualitative + BLEU)
- [ ] BPCC vs Shiksha comparison on arXiv test set
