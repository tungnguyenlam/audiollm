# Vietnamese datasets (survey, 2026-10-29)

This survey asks whether we can fine-tune the log-mel → `Linear` projector → LFM2.5 model to understand Vietnamese,
and which data we would use. **Nothing here has been downloaded into `data/`, and no code supports Vietnamese yet.**
It uses the same label tiers (S/A/B/C) as [`DATASETS.md`](DATASETS.md#1-how-the-label-quality-tiers-are-defined).

**How it was checked:** "Verified" means we hit the HF API, downloaded files or ran code on 2026-10-29. "Card" means we
only read the dataset card or paper. For eight open sets we sampled 1–2 shards each and measured:

- clip length;
- text format;
- speaking rate;
- a 20-clip CER spot-check against `vinai/PhoWhisper-small`.

The CER check uses lowercase text with punctuation stripped, and it is only a rough check: PhoWhisper may have been
trained on some of these sets, and on a CER above about 20% it is often the model that is wrong. Raw output is in the
session scratchpad, which is not kept.

## Contents
1. [Will it work? The tokenizer problem](#1-will-it-work-the-tokenizer-problem)
2. [ASR corpora: ranking](#2-asr-corpora-ranking)
3. [Per-dataset notes](#3-per-dataset-notes)
4. [Evaluation sets and reference numbers](#4-evaluation-sets-and-reference-numbers)
5. [Audio understanding beyond ASR](#5-audio-understanding-beyond-asr)
6. [Scoring Vietnamese WER](#6-scoring-vietnamese-wer)
7. [Suggested plan](#7-suggested-plan)
8. [Sources](#8-sources)

## 1. Will it work? The tokenizer problem

The LFM2/LFM2.5 model cards list en, ar, zh, fr, de, ja, ko and es. They do not list Vietnamese. The tokenizer falls
back to bytes for diacritics: "Hôm nay trời" becomes `H`, `Ã´`, `m`, `Ġn`, `ay`, `Ġtr`, `á»`, `Ŀ`, `i`, ...

Tokens per word, measured with `AutoTokenizer` and no special tokens:

| tokenizer (vocab) | short vi (12 words) | news vi (35 words) | English (17 words) |
|---|---|---|---|
| **LFM2.5-230M / 1.2B, LFM2-2.6B (64K)** — current backbone and paper-scale target | **3.00** | **2.97** | 1.12 |
| SmolLM2 (49K) | 3.17 | 3.00 | 1.12 |
| Granite-3.3-2B (49K) | 2.42 | 2.20 | 1.18 |
| Qwen3-0.6B/1.7B, Qwen2.5, SeaLLMs-v3-1.5B, Sailor2-1B (152K) | 1.42 | 1.17 | 1.12 |
| Gemma-3 270M/1B, Gemma-3n (262K) | 1.33 | 1.11 | 1.12 |
| Llama-3.2-1B (128K) | 1.33 | 1.11 | 1.12 |
| LFM2.5-8B-A1B (125K, tokenizer expanded in place) | 1.33 | 1.14 | 1.12 |

On the current backbone, Vietnamese costs about 2.7× the decode steps of English. The byte pieces are embeddings the
model rarely trained on, so the LLM's language knowledge helps little. Expect `MAX_TOKENS_PER_SEC` to need about 3×
its English value.

Inference, not measured: an encoder-free model starts with no acoustic knowledge. SLAM-ASR-style systems, which have
a Whisper encoder, needed about 200 h of a new language to match Whisper (Fong et al. 2025), so we will probably
need thousands of hours.

**Options, cheapest first:**
- **(a) Keep the tokenizer and measure the cost.** Run LFM2.5 with byte fallback as a baseline ablation.
- **(b) Switch to a backbone that covers Vietnamese.** Qwen3-0.6B, Gemma-3-270M/1B or Sailor2-1B. This is the
  usual choice in published work (Sailor, SeaLLM, EuroLLM).
- **(c) Expand the vocabulary, Liquid-style** ([arXiv 2607.15232](https://arxiv.org/abs/2607.15232)):
  - continue the BPE merges;
  - initialise each new row as the mean of its sub-token embeddings;
  - warm up the embeddings on text only;
  - then train the embeddings and `lm_head`, not only LoRA.

## 2. ASR corpora: ranking

Hours are for Vietnamese.

**Access** means:
- **open**: downloads with our HF account today;
- **gated-auto**: accept the terms on HF and you are in immediately;
- **gated-manual**: someone reviews your request;
- **meta**: the dataset holds only metadata; you fetch the audio yourself.

**NC** = non-commercial.

| tier | dataset (HF id) | hours | domain | how transcripts were made | text format | access / licence | verdict |
|---|---|---|---|---|---|---|---|
| S (dev/test) / C (train) | **GigaSpeech 2 vi** (`speechcolab/gigaspeech2`, `data/vi/`) | 6,039 refined (7,324 raw); dev 10.2, test 11.0 | YouTube, many domains | Train: Whisper-large-v3 pseudo-labels → MMS forced alignment → 3 rounds of noisy-student filtering. Dev/test: professional transcription at >97% accuracy, every item checked | UPPERCASE, no punctuation, numbers as words | gated (HF terms + a Google Form); research-only NC despite the apache-2.0 tag. ~824 GB train | **Main training source if we go big. Its test set is the main benchmark.** |
| A | **PhoAudiobook** (`thivux/phoaudiobook`) | 940 + 554 augmented short = 1,494 | audiobooks, 710 speakers | Book text aligned to audio (VinAI, ACL 2025) | lowercase, punctuated, numbers as words | gated-auto, **downloads fine for us**; research only, no redistribution. 167 GB | **Cleanest labels measured: 0.9% CER.** The first set to train on (the "LibriSpeech" of this plan) |
| S | **VLSP 2020 (VinAI 100 h)** (`doof-ferb/vlsp2020_vinai_100h`) | 100 | conversational and reading, varied | Human transcription (card: 96% accuracy) | lowercase, no punctuation, numbers as words; foreign words spelled phonetically | open, CC-BY-4.0 (unofficial mirror). 11.6 GB | Train. 2.2% CER |
| A/S | **LSVSC** (`doof-ferb/LSVSC`) | 100 | news, varied topics | Human (MDPI Electronics 2024) | cased 79%, punctuated 37%; **numerals and spelled-out numbers mixed** | open, CC-BY-4.0 (mirror). 11.6 GB | Train after converting digits to words. 6.2% CER (mostly number format). Has topic/gender/dialect/emotion/age columns |
| S | **ViMD** (`nguyendv02/ViMD_Dataset`) | 102.6 (81 / 10 / 11) | broadcast news from 63 provinces | Human (EMNLP 2024) | cased, punctuated, fillers kept, numbers as words | open, **CC-BY-NC-ND** | Dialect coverage. 48/44.1 kHz, so resample; clips up to 30 s. 12% CER is PhoWhisper struggling with dialects, not bad labels |
| S | **FOSD** (`doof-ferb/fpt_fosd`; Mendeley) | ~30 (the mirror card says 100, which contradicts the source) | read and spontaneous | Human (FPT, 2018) | — | open; FPT licence is MIT-like and allows commercial use | Small extra set. Its clips are also in the VietAudio leaderboard eval |
| B/C | **VietSpeech** (`NhutP/VietSpeech`) | ~1,100 (1.03M clips) | social media, 3 accents | Not documented; PhoASR groups it with sets labelled by an ASR model | lowercase, no punctuation, numbers as words | gated-auto, apache-2.0. 131 GB | Filter before use: only about 53% passed PhoASR's WER < 5% check against PhoWhisper |
| B/C | **Bud500** (`linhtran92/viet_bud500`) | ~500 (634k clips); val/test 5.5 h each | YouTube, all regions | Not documented | lowercase, no punctuation; chunks cut mid-phrase | gated-auto; the licence is inconsistent (CC-BY-NC-SA tag, Apache notice in the card). 53 GB | Filter: only about 46% passed PhoASR's filter |
| C | **LEMAS vi** (`LEMAS-Project/LEMAS-Dataset-train`, `train/vi/`) | 6,600 (6.4M clips) | not stated | Aligned audio–text pairs, kept if mean alignment confidence > 0.2–0.5; word timestamps included | normalised `align.txt` | open, CC-BY-4.0 | **Unexplored and large.** Audit a shard before trusting it. Has a 500-utterance eval set |
| C | **viVoice** (`capleaf/viVoice`) | ~1,000 (888k clips) | 186 YouTube channels, denoised, 24 kHz | Not stated; a manual check found 1.8% of sentences with the wrong word count (substitutions ignored) | cased, punctuated | gated (card says institutional email and 2–3 days of review; API says auto); CC-BY-NC-SA | Built for TTS. PhoASR kept only 37% of it. Low priority |
| C | **YODAS2 vi** (`espnet/yodas2`) | ~4.8k manual-caption + ~10k automatic-caption hours (whole videos, before segmentation) | YouTube | User or automatic YouTube captions | mixed | open, CC-BY-3.0 | Only with heavy filtering |
| C | **VietSuperSpeech** (`thanhnew2001/VietSuperSpeech`) | **245 h actual** (the card says 103 h; the paper says 267 h) | vietcetera podcasts | Zipformer pseudo-labels | ALL CAPS, no punctuation | open, no licence tag | **Do not use as is.** 16.8% of rows contain nonsense English ("HAD YOU GHE MICHEL US TOP LEVINING…"); 51.6% CER against PhoWhisper |
| S | **VietMed** (`leduckhai/VietMed`) | 16 labelled (+1,000 h medical / 1,200 h general unlabelled) | medical | Human | lowercase; 0.5% have `[dạ]`-style tags | open, MIT | **8 kHz audio.** Use as a medical-domain eval only |
| S | **ViMedCSS** (`tensorxt/ViMedCSS`) | 31 (train 24.3 / val 3.6 / test 3.4 / hard 1.4) | medical Vietnamese–English code-switching | Human (LREC 2026) | — | open, CC-BY-4.0 | Niche eval |
| S | **WorldSpeech vn_vi** (`disco-eth/WorldSpeech`) | 726 | Radio Free Asia | Human (broadcaster transcripts) | — | meta, CC-BY-NC-4.0 | Good labels, but the audio must be fetched from source URLs |
| B | **VIVOS** (`AILAB-VNUHCM/vivos`) | 15 | read prompts | Prompt text | ALL CAPS | open, CC-BY-NC-SA | Small. Prompts repeat across speakers (12% duplicates) |
| B | **FLEURS vi_vn** (`google/fleurs`) | ~12 | read Wikipedia | Prompt text | cased, punctuated, digits | open, CC-BY-4.0 | **Eval only.** Score on its normalised column |
| B | **Common Voice vi** | 7.7 validated / 24 recorded | volunteers reading prompts | Votes | cased | Mozilla Data Collective (CV 27.0, CC0) | Eval only; tiny |
| — | DNU-ViSpeech (`TrangLe2108/DNU-ViSpeech`) | 75.6 | read, 26 speakers | — | — | CC-BY-NC-4.0 | Not audited |
| — | Granary-v2 vi (`nvidia/Granary-v2`, `vi/`) | — | relabelled FLEURS and MCV26 | Pseudo-labels | — | manifests only | Not useful (it relabels sets we already have) |
| — | PhoASR (Qualcomm, arXiv 2603.14779) | ~500, cleaned aggregate | — | Existing corpora filtered at WER < 5% against PhoWhisper | — | github.com/qualcomm-ai-research/PhoASR; no HF id found | Copy its filtering recipe |
| ✗ | SpeechDataAI conversational | 2,000 | — | — | — | commercial | Not open |

Emilia and Granary v1 do not include Vietnamese.

## 3. Per-dataset notes

These are measured on samples, unless marked as coming from the card.

| set | sample rate | mean / p5 / p95 clip (s) | median words/s | notes |
|---|---|---|---|---|
| VLSP2020 | 16k | 4.4 / 1.2 / 9.1 | 3.6 | 1.9% of clips < 1 s; 0.5% duplicate texts. PhoWhisper probably trained on it, so its 2.2% CER is optimistic |
| LSVSC | 16k | 6.2 / 2.1 / 11.3 | 4.2 | Mixed "25,5 đô la" and "bảy triệu năm trăm nghìn"; occasional misplaced commas |
| ViMD | 48k/44.1k | 17.9 / 7.4 / 28.2 | 3.5 | Clips are long (≤ 30.0 s). Our training cap may need to change |
| PhoAudiobook | 16k | 3.5 / 1.1 / 11.2 | 3.7 | 2.5% of clips < 1 s (drop them); some fragment clips |
| VietMed | **8k** | 6.3 / 5 / 8 | 3.8 | Band-limited; durations are whole seconds |
| VIVOS | 16k | 4.4 / 2.3 / 9.5 | 3.0 | Read speech |
| FLEURS vi | 16k | 11.4 / 6.1 / 18.7 | 2.6 | Text has `km<sup>2</sup>`, `$`, `°`; it also ships a normalised column |
| VietSuperSpeech | 16k | 13.1 / 10.4 / 15.0 | 3.8 | 0.8% of clips below 0.5 words/s, a sign of misalignment |

**GigaSpeech 2 vi format** (from the GitHub README): NFKC, uppercase, no punctuation, numbers as words, disfluencies
(à, ờ) kept. Dev/test examples from the mirror `doof-ferb/gigaspeech2_vie`: "à riêng nhóm hai thì à các con sẽ cố
định cho bánh xe nó chặt vào một tí xíu nha con".

**Access steps:**
- **Gated-auto sets** (VietSpeech, Bud500, PhoAudiobook): click "accept" on HF, logged in as `tungnguyenlam`.
- **GigaSpeech 2:** the HF form asks you to confirm you have also filled in SpeechColab's Google Form (the link is on
  the GitHub page).
- **viVoice:** the card asks for an institutional email.

## 4. Evaluation sets and reference numbers

**What to score on:**
- **GigaSpeech 2 vi test (11 h, professional transcripts):** the main benchmark.
- **FLEURS vi, Common Voice vi and VIVOS test:** read speech.
- **ViMD test:** dialects.
- **VLSP 2020:** conversational.
- **VietMed / ViMedCSS:** domain stress tests.

**VietAudio-team leaderboard** ([Space](https://huggingface.co/spaces/VietAudio-team/Vietnamese-ASR-Leaderboard);
data `VietAudio-team/Vietnamese-asr-leaderboard`, 212 h across 9 configs):
- The data has **audio and anonymised ids only**; the transcripts are private. You score by uploading `id,
  transcription` CSVs.
- Its fosd, vivoice and vivos configs look like random subsets of the whole corpus, so training on those corpora
  probably leaks into this eval.
- It reports two metrics:
  - **O-WER:** cased text, punctuation split into separate tokens;
  - **N-WER:** lowercase, punctuation removed.

**Published WERs** (normalisation differs between sources, so compare within a row source only):

| system | GS2-vi test | CV-vi | FLEURS-vi | VIVOS | other |
|---|---|---|---|---|---|
| PhoWhisper-large (844 h fine-tune) | – | 8.14 | – | 4.67 | VLSP T1 13.75 |
| Whisper large-v3 | 17.94 | 13.74 | 8.59 | – | – |
| Zipformer, GigaSpeech2 recipe (152M) | 12.83 | 14.43 | 11.59 | – | – |
| VietASR, 68M Zipformer (SSL + 50 h labelled) | 7.68 | 11.46 | 10.96 | – | – |
| Azure / Google USM | 11.86 / 13.28 | 10.21 / 12.46 | 11.88 / 11.75 | – | – |
| Qwen3-ASR-1.7B / 0.6B (tech report) | – | – | 5.55 / 8.52 | – | – |
| Leaderboard N-WER: Zipformer-30M, 6,000 h | 6.54 | 3.97 | – | 3.61 | Bud500 1.93, VLSP 7.26 |
| Leaderboard N-WER: Qwen3-ASR-1.7B | 8.24 | 8.77 | – | 6.11 | Bud500 4.89, VLSP 9.88 |
| Speech-LLM: FastConformer + Sailor-0.5B + LoRA (community; first 300 utterances only) | – | 21.6 | 18.4 | 9.9 | Bud500 10.4, VLSP 30.5 |

**What counts as competitive:** about 10% normalised WER on GS2-vi and CV-vi, and under 6% on FLEURS-vi.

**Leaderboard N-WER averages** (synthetic sets / natural sets):

| model | synthetic | natural |
|---|---|---|
| Zipformer-30M | 3.79 | 6.77 |
| PhoWhisper-medium | 4.35 | 7.61 |
| Qwen3-ASR-1.7B | 7.44 | 9.84 |
| wav2vec2-base-vi-250h | 13.51 | 14.89 |

## 5. Audio understanding beyond ASR

There is **no open, human-recorded Vietnamese speech-instruction training set.** Most of what exists is small,
synthesised with TTS, or NC.

| task | dataset | size | labels / audio | access | note |
|---|---|---|---|---|---|
| eval suite | **SEA-SpeechBench** (`MERaLiON/sea_audiobench_datasets_{ASR,SQA,SpR,ST,TCQ,TLoc,PQA}`) | ~30 items per config | Human / curated | open, mostly CC-BY-NC-ND | **Best ready-made Vietnamese eval.** Vietnamese configs: SQA (yodas2), temporal QA, speaker recognition (VoxVietnam), gender/age (CV21), ST (FLEURS), ASR. No emotion |
| spoken QA | **ViSQA** ([Figshare](https://figshare.com/articles/dataset/ViSQA_v1_0_0/29493149)) | 10,441 train / 1,507 test QA pairs (+ a noisy test set) | Human QA (from UIT-ViQuAD); **audio is Google TTS** | CC-BY-4.0, 1.14 GB zip | Extractive QA; single-voice TTS |
| SLU (intent/slot) | **Speech-MASSIVE vi-VN** (`FBK-MT/Speech-MASSIVE`; test in `-test`) | dev + few-shot train (115) | Crowd-read MASSIVE text, human labels | CC-BY-NC-SA | The only real Vietnamese spoken-SLU set |
| emotion | **ViSEC** (mirrors `Dennis290/ViSEC`, `Vinh727/HusTep_ViSEC`; the official repo is gone) | 5,280 utterances, 4 emotions | Human | CC-BY-4.0 | Best small SER set |
| emotion | `qdovan03/vietspeech-emo` | 20k | Model pseudo-scores | no licence | Weak |
| dialect / gender | ViMD (63 provinces); `hr16/ViSpeech-Gender-Dialect-Classification` (19.7k, 3 regions) | — | Human | NC-ND / Apache | — |
| ST En→Vi | **PhoST** (`vinai/PhoST`) | 508 h TED | Human subtitles | gated-auto, research only | English audio only |
| ST Vi→En | FLEURS vi; `anquachdev/PhoMT-en-vi-speech` (1,228 h vi, **TTS**); `huybik/en-vi-speech` (~9.4k h, TTS voice-cloned) | — | Synthetic | Apache / unclear | Synthetic only, apart from FLEURS |
| speech chat | `jan-hq/mixed-speech-instruction-ichigo-tokens-vi-en` | 2.9M conversations | TTS, **stored as WhisperVQ tokens, no waveforms** | no licence | Needs re-synthesis |
| smoke test | `SeaLLMs/SeaBench-Audio` | ~15 items per task | — | Apache | Too small |
| paper only | SentiV (model repo only, no audio), VlogQA (text only), PhoATIS (text only), VNEMOS, VESC | — | — | — | Not usable |

## 6. Scoring Vietnamese WER

There is no common standard. Common practice:
- **Unit:** syllable-level WER (Vietnamese separates syllables with spaces), with CER rarely reported.
- **Text:** NFC, lowercase, punctuation stripped.
- **Numbers:** in **spoken form** (GigaSpeech2, VLSP, the leaderboard).

`benchmark.py`'s Whisper English normaliser is not suitable. For Vietnamese scoring, apply the same steps to both
reference and hypothesis:

1. `unicodedata.normalize("NFC")`.
2. **Unify tone placement** ("hoà"→"hòa", "thuỷ"→"thủy"): NFC alone does not do this. `underthesea`'s
   `text_normalize` does (tested with underthesea 9.5.0). It also splits off punctuation, which is harmless here.
3. **Digits, dates and currency → words:** `vietnormalizer` (tested, works). Do **not** use `num2words(lang="vi")`:
   it writes "lẻ", while the leaderboard writes "không trăm". `vinorm` does not import on Python 3.12.
   `PhoTextNormalization` (mBART-large, 0.6B) is too heavy for a scoring loop.
4. Lowercase, strip punctuation, collapse whitespace.
5. **Map spelled-number variants:** linh→lẻ, ngàn→nghìn, and "mươi bốn"→"mươi tư".

**Training targets:** use spoken form, lowercase. Turn written-form corpora (FLEURS, LSVSC; ViMD is already spoken
form) into spoken form once, offline. NeMo-text-processing has Vietnamese TN/ITN grammars if we later want written
output.

## 7. Suggested plan

Not started. This is a starting point for later.

1. **Choose the backbone first.** It is a design decision. Run the same ~100 h experiment twice, once on LFM2.5-230M
   (byte fallback) and once on a Vietnamese-capable backbone of similar size (Qwen3-0.6B or Gemma-3-270M).
   - Training data: PhoAudiobook slice + VLSP2020 + LSVSC.
   - Evaluation: GS2-vi test (needs the gate), FLEURS-vi and ViMD test.
   - The gap between the two tells us what the tokenizer costs.
2. **Stage 1 data (~300–400 h, labels A/S, all downloadable today, about 50 GB):**
   - PhoAudiobook subset, 16 kHz;
   - VLSP2020;
   - LSVSC;
   - ViMD train (NC-ND);
   - FOSD.
3. **Stage 2 (thousands of hours, label C):**
   - GigaSpeech 2 vi refined, after the gate (~824 GB, larger than our 100 GB `data/` budget, so stream it or take a
     subset);
   - LEMAS vi, after an audit;
   - VietSpeech and Bud500, filtered the PhoASR way (keep clips at WER < 5% against a strong teacher).
4. **Understanding tasks:** only after ASR works. Start with SEA-SpeechBench vi for eval, and ViSQA / ViSEC /
   Speech-MASSIVE for small fine-tunes.

## 8. Sources

**Corpus papers and repos:**
- GigaSpeech 2: <https://arxiv.org/abs/2406.11546>, <https://github.com/SpeechColab/GigaSpeech2>
- PhoAudiobook: <https://arxiv.org/abs/2506.01322>
- ViMD: EMNLP 2024 (HF card)
- LSVSC: <https://www.mdpi.com/2079-9292/13/5/977>
- VietMed: <https://arxiv.org/abs/2404.05659>
- VietSuperSpeech: <https://arxiv.org/abs/2603.01894>
- PhoASR, "Vietnamese ASR: A Revisit": <https://arxiv.org/abs/2603.14779>
- FOSD: <https://data.mendeley.com/datasets/k9sxg2twv4/4>
- Common Voice vi: <https://mozilladatacollective.com/datasets/cmu5vci3w00agnq07hy4nbauv>
- Dataset catalogue: <https://github.com/lienlt10/awesome-vietnamese-speech-datasets>

**Models, leaderboards and adaptation:**
- PhoWhisper: <https://arxiv.org/abs/2406.02555>
- VietASR: <https://arxiv.org/abs/2505.21527>
- Qwen3-ASR: <https://arxiv.org/abs/2601.21337>
- VietAudio leaderboard: <https://huggingface.co/spaces/VietAudio-team/Vietnamese-ASR-Leaderboard>
- Low-resource SLAM-ASR (Fong et al. 2025): <https://arxiv.org/abs/2508.05149>
- In-place tokenizer expansion (Liquid): <https://arxiv.org/abs/2607.15232>

**Understanding datasets:**
- ViSQA: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12795348/>
- SEA-SpeechBench: <https://zwenyu.github.io/SEA-SpeechBench/>
- PhoST: <https://github.com/VinAIResearch/PhoST>
- SeaLLMs-Audio: <https://github.com/DAMO-NLP-SG/SeaLLMs-Audio>
- VlogQA: <https://github.com/sonlam1102/vlogqa>
