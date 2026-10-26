# Datasets

This doc covers the English ASR corpora and the audio-understanding datasets that matter for this project, i.e. a
log-mel → `Linear` projector → LFM2.5 model. For each dataset it gives:

- a tier for how good its **labels** are (not the audio, and not the size);
- whether we can get it, and whether it is already on disk;
- a short summary;
- a citation (BibTeX is at the end of the doc).

`download_corpora.py` fetches the corpora (`--list` shows what's there; `--only a,b` fetches particular ones).
`prepare_data.py` unpacks the LibriSpeech splits. `data/` is capped at about **100 GB**, so only the high-quality
corpora are downloaded by default. Everything else is in the registry but switched off.

> [!IMPORTANT]
> `train.py` still trains on LibriSpeech `train-clean-100` only. The other corpora are on disk, but no data loader
> for them exists yet. Numbers marked "~" are approximate, and corpus releases differ, so check the paper or the
> dataset card before quoting a number.

## Contents
1. [How the label-quality tiers are defined](#1-how-the-label-quality-tiers-are-defined)
2. [ASR corpora: ranking](#2-asr-corpora-ranking)
3. [ASR corpora: summaries](#3-asr-corpora-summaries)
4. [Audio understanding: ranking](#4-audio-understanding-ranking)
5. [Audio understanding: summaries](#5-audio-understanding-summaries)
6. [What's on disk](#6-whats-on-disk)
7. [References (BibTeX)](#7-references-bibtex)

## 1. How the label-quality tiers are defined

| tier | meaning | typical failure mode |
|---|---|---|
| **S** | Humans transcribed or labelled the audio itself, as their job, with checking or multiple passes | Style conventions (verbatim vs. clean read) differ between corpora |
| **A** | The text was written by people (a book, captions, official records), then automatically aligned to the audio and filtered. Or the label is true by construction (a prompted command, an elicited sound), or several annotators agreed | Reader ad-libs; captions that aren't verbatim; filtering leftovers |
| **B** | Crowd-sourced, a single pass or vote-validated. Or human text aligned at very large scale with no human check | Misreadings, missing events, noisy prompts |
| **C** | Weak or automatic: pseudo-labels from an ASR model, scraped subtitles or tags, captions or Q&A written by an LLM | Hallucinations, paraphrases, timing drift; inherits the teacher model's mistakes |

**Text format matters for an LLM decoder.** Corpora with cased, punctuated text (LibriTTS, SPGISpeech, Libriheavy,
Common Voice, FLEURS) fit a pretrained LLM's natural writing style. Normalised text (LibriSpeech is uppercase with
no punctuation; TED-LIUM, VoxPopuli and MLS are lowercase) has to be scored with the Whisper normaliser, which
`benchmark.py` already uses. Never mix the two formats in one training target without a prompt that says which
style is wanted.

## 2. ASR corpora: ranking

Within each tier, the rows are roughly best first. "Access" means: **open** = open on HF; **gated** = you have to
accept terms on HF; **LDC** = paid licence; **eval** = only test/dev splits exist.

| tier | dataset | English hours | domain | how the transcript was made | text format | access | notes |
|---|---|---|---|---|---|---|---|
| S | Switchboard-1 | ~260 h | telephone conversations | LDC transcribers, plus a later re-transcription by Mississippi State | lowercase, with disfluency marks | LDC | Classic conversational benchmark; 8 kHz |
| S | WSJ (CSR-I/II) | ~80 h | read newspaper text | Prompt text, checked against the recordings | uppercase, normalised | LDC | Very clean, narrow domain |
| S | AMI | ~100 h | meetings (headset + far-field) | Manual transcription of the meetings | cased + punctuated (raw) | open | **On disk (IHM).** Has train/dev/test |
| S | ICSI Meetings | ~72 h | research meetings | Manual verbatim transcription | cased (raw) | LDC / open audio | Lots of overlapping speech |
| S | CHiME-5/6 | ~50 h | dinner parties at home | Manual transcription from binaural recordings | normalised | licence from organisers | Very hard far-field audio |
| S | SPGISpeech | ~5,000 h | earnings calls | Professional transcription, fully formatted | cased, punctuated, numerals | gated | Best large source of *formatted* text; non-commercial |
| S | Earnings-21 / 22 | 39 h / 119 h | earnings calls, accents | Professional transcription | cased + punctuated | open, **eval** | Test only (used in our benchmark) |
| S | EdAcc | ~40 h | accented video calls | Manual transcription | cased + punctuated | open, **eval** | **On disk.** Dev/test only: an accent stress test |
| A | LibriSpeech | 960 h | read audiobooks | Book text aligned to audio and filtered | UPPERCASE, no punctuation | open | **On disk: 100 + 360 h, all dev/test.** 500 h "other" is optional |
| A | LibriTTS / LibriTTS-R | ~585 h | read audiobooks | LibriSpeech material re-split at sentence boundaries, keeping the original book text | cased + punctuated | open | Optional (~100 GB at 24 kHz) |
| A | Fisher English | ~2,000 h | telephone conversations | Quick transcription by LDC | lowercase | LDC | Faster, less careful style than Switchboard |
| A | TED-LIUM 3 | ~452 h | TED talks | TED captions aligned to audio | lowercase | open (CC BY-NC-ND) | Captions aren't fully verbatim |
| A | VoxPopuli (en, transcribed) | ~543 h | European Parliament | Official session records aligned and filtered | lowercase (raw text cased) | open | Optional (~99 GB on HF) |
| A | VCTK | ~44 h | read prompts, 110 speakers | Prompt text | cased + punctuated | gated mirror / CSTR | Studio quality, small |
| B | GigaSpeech | 10,000 h (+33k h unlabelled) | podcasts, YouTube, audiobooks | Existing human transcripts or captions, automatically aligned and filtered; dev/test checked by humans | uppercase + punctuation tags | gated | Very varied; train labels not hand-checked |
| B | Libriheavy | ~50,000 h | read audiobooks | Original book text aligned to Libri-Light audio by an ASR-based pipeline | cased + punctuated | open | Real text; alignment never checked by hand |
| B | MLS (English) | ~44,500 h | read audiobooks | Book text matched to audio automatically | lowercase | open | Optional (~705 GB) |
| B | Common Voice (en) | thousands of hours (varies by release) | volunteers reading prompts | Prompt text; clips kept after volunteers vote on them | cased + punctuated | Mozilla Data Collective | Moved off HF in 2025; uneven microphones |
| B | FLEURS (en_us) | ~12 h | read Wikipedia sentences (FLoRes) | Prompt text | cased + punctuated | open | **On disk.** Small; good for sanity checks |
| C | People's Speech | ~30,000 h | public-domain web audio | Existing captions aligned automatically | mixed | open | Noisy; optional (~400 GB) |
| C | YODAS | very large, 100+ languages | YouTube | User-uploaded and automatic subtitles | mixed | open | Paraphrased or shifted subtitles |
| C | Emilia | ~47k h English | in-the-wild speech | Pseudo-labels from an ASR model | cased + punctuated | open (CC BY-NC) | Built for TTS; inherits ASR mistakes |
| C | Granary | very large, 25 languages | web speech | Pseudo-labels with multi-stage filtering | cased + punctuated | open | Labels are fully automatic |
| — | Libri-Light | ~60,000 h | read audiobooks | **No transcripts** (plus 10 h / 1 h / 10 min labelled subsets) | — | open | Self-supervised learning; the audio source for Libriheavy |
| — | ESB | — | 8 test sets | Benchmark spec | — | — | How our `benchmark.py` sets are defined |

## 3. ASR corpora: summaries

**LibriSpeech** (Panayotov et al., ICASSP 2015 [`panayotov2015librispeech`]). About 1,000 h of 16 kHz read English
from LibriVox audiobooks. Each segment is aligned to its Project Gutenberg text, and segments with poor alignment
were dropped. The training data comes in three splits: `train-clean-100`, `train-clean-360` and `train-other-500`.
Speakers were sorted into "clean" and "other" by how well a baseline model recognised them. There are matching
dev and test sets for both, and the licence is CC BY 4.0. Transcripts are uppercase with no punctuation. It's our
in-domain training and validation data, and `train-clean-360` and `dev-other` are now on disk next to the
original 100 h.

**LibriTTS / LibriTTS-R** (Zen et al., Interspeech 2019 [`zen2019libritts`]; Koizumi et al., Interspeech 2023
[`koizumi2023librittsr`]). About 585 h at 24 kHz, built from the LibriSpeech source material but cut at sentence
breaks. It keeps the **original book text with casing and punctuation**, alongside a normalised version.
LibriTTS-R is the same corpus with the audio cleaned up by a speech-restoration model. Licence CC BY 4.0. It's
the easiest way to train on cased, punctuated targets without leaving the LibriSpeech domain. Its speakers and
material overlap LibriSpeech, so keep the official splits to avoid leaking test data.

**Libri-Light** (Kahn et al., ICASSP 2020 [`kahn2020librilight`]). About 60k h of **unlabelled** LibriVox audio
for self-supervised and semi-supervised ASR. It comes with small labelled subsets (10 h, 1 h, 10 min) for
low-resource evaluation. It is the audio source for Libriheavy.

**Libriheavy** (Kang et al., ICASSP 2024 [`kang2024libriheavy`]). About 50k h of Libri-Light audio paired with
the original book text. Casing, punctuation and preceding text are kept, and an automatic ASR-based pipeline did
the alignment. It comes in small, medium and large training subsets with book-disjoint dev/test sets. It's the
most natural next scale-up after LibriTTS for an LLM decoder, but no human checked the alignments.

**TED-LIUM 3** (Hernandez et al., SPECOM 2018 [`hernandez2018tedlium3`]). About 452 h from 2,351 TED talks, with
TED's captions aligned to the audio. It has a redesigned split for speaker-adaptation experiments and is licensed
CC BY-NC-ND 3.0. The speech is prepared but spontaneous-sounding. Captions sometimes drop disfluencies.

**Common Voice** (Ardila et al., LREC 2020 [`ardila2020commonvoice`]). Volunteers read prompt sentences, and
other volunteers vote clips up or down. The transcript is the prompt text. It is many-language, CC0 and huge, but
microphones and accents are uneven, and some misread clips get through validation. Mozilla took the data off HF
in 2025; it's now distributed through the Mozilla Data Collective.

**VoxPopuli** (Wang et al., ACL 2021 [`wang2021voxpopuli`]). Recordings of European Parliament sessions,
2009–2020. It has about 400k h unlabelled in 23 languages, plus transcribed subsets made by aligning the official
session records; English has about 543 h. Licence CC0. Speakers are largely non-native and the speech is formal.
The official records are edited, so they don't always match what was said word for word.

**AMI** (Carletta et al., MLMI 2005 [`carletta2005ami`]). About 100 h of meetings, many of them role-played design
meetings, recorded with headsets (IHM) and microphone arrays (SDM/MDM). Transcription was manual and the licence
is CC BY 4.0. It's our in-domain source for conversational speech; the IHM version is on disk.

**ICSI Meeting Corpus** (Janin et al., ICASSP 2003 [`janin2003icsi`]). About 72 h of real research-group meetings
at ICSI Berkeley, with close-talking and table microphones and manual verbatim transcripts. It's a useful
complement to AMI because the meetings weren't scripted.

**CHiME-5 / CHiME-6** (Barker et al., Interspeech 2018 [`barker2018chime5`]; Watanabe et al., CHiME 2020
[`watanabe2020chime6`]). Twenty dinner parties in real homes, recorded with binaural mics and distant arrays and
transcribed by hand. CHiME-6 added array synchronisation and a diarisation track. It's one of the hardest
far-field conversational benchmarks.

**GigaSpeech** (Chen et al., Interspeech 2021 [`chen2021gigaspeech`]). 10k h of transcribed audiobooks, podcasts
and YouTube, plus about 33k h of audio without transcripts. Training subsets run from XS (10 h) to XL (10k h).
The transcripts came from existing human-written sources and were aligned and filtered automatically. The
dev/test sets were checked by humans. Punctuation is kept as tags such as `<COMMA>`. It's gated on HF and our
token doesn't have access yet.

**SPGISpeech** (O'Neill et al., Interspeech 2021 [`oneill2021spgispeech`]). About 5,000 h of earnings calls
transcribed professionally in a **fully formatted** style: casing, punctuation, numerals, currencies. It's the
best large source for training a model to write formatted text directly. Gated, non-commercial licence.

**Earnings-21 / Earnings-22** (Del Rio et al., Interspeech 2021 [`delrio2021earnings21`]; Del Rio et al., arXiv
2022 [`delrio2022earnings22`]). Earnings-21 is 39 h and Earnings-22 is 119 h of real earnings calls, focused on
named entities and on speakers from many countries. Rev transcribed them professionally, and the licence is
CC BY-SA 4.0. They are test-only, and Earnings-22 is one of our benchmark sets.

**The People's Speech** (Galvez et al., NeurIPS Datasets & Benchmarks 2021 [`galvez2021peoplesspeech`]). About
30k h of CC-licensed and public-domain web audio, with existing transcripts aligned automatically. Its
commercial-friendly licence is the main draw; the labels are noisy, even in the "clean" subset.

**MLS** (Pratap et al., Interspeech 2020 [`pratap2020mls`]). Multilingual LibriVox read speech in 8 languages; the
English part is about 44.5k h. The book text was matched to the audio automatically. Transcripts are lowercase
with no punctuation. It's a large, steady read-speech source, but for an LLM decoder Libriheavy's formatted text
is usually the better pick.

**FLEURS** (Conneau et al., IEEE SLT 2022 [`conneau2023fleurs`]). Read versions of the FLoRes sentences in 102
languages, about 12 h each. The prompt text is the transcript, with casing and punctuation. It's tiny, but useful
as a clean, formatted dev set. `en_us` is on disk.

**YODAS** (Li et al., ASRU 2023 [`li2023yodas`]). A very large multilingual YouTube corpus from Creative
Commons–licensed videos. It has a "manual" subset (user-uploaded subtitles) and an "automatic" subset (generated
subtitles). Subtitles are often paraphrased or out of sync, so they need filtering before use.

**Emilia** (He et al., IEEE SLT 2024 [`he2024emilia`]). About 101k h of in-the-wild speech in six languages,
roughly 47k h of it English. It was built for speech *generation*: the audio was cut by an automatic pipeline and
transcribed by an ASR model. Licence CC BY-NC 4.0. Training ASR on it partly distils the teacher model, mistakes
included.

**Granary** (Koluguri et al., Interspeech 2025 [`koluguri2025granary`]). Large-scale ASR and speech-translation
data for 25 European languages. It was made by pseudo-labelling existing open audio, followed by several stages
of filtering for hallucinations and language ID, and punctuation restoration. It offers high volume, but the
labels are entirely automatic.

**EdAcc** (Sanabria et al., ICASSP 2023 [`sanabria2023edacc`]). About 40 h of conversations over video calls
between speakers with a wide range of first and second languages, transcribed by hand. It only has dev and test
splits, so it's for evaluating accent robustness, not for training. It is on disk.

**VCTK** (Yamagishi et al., 2019 [`yamagishi2019vctk`]). 110 English speakers with varied accents, each reading
about 400 newspaper and elicitation sentences in a studio, about 44 h in total. Licence CC BY 4.0. Its label
quality is high but it's small and read speech; it's mostly used for TTS.

**WSJ** (Paul & Baker, 1992 [`paul1992wsj`]). The DARPA CSR read-speech corpus of Wall Street Journal text, about
80 h across WSJ0 and WSJ1. The text is carefully controlled and it's sold by LDC. It's historically important but
narrow.

**Switchboard-1** (Godfrey et al., ICASSP 1992 [`godfrey1992switchboard`]). About 2,400 telephone conversations
between strangers on assigned topics, roughly 260 h at 8 kHz, transcribed by humans. Mississippi State later
re-transcribed it to fix errors. The standard conversational benchmark; sold by LDC.

**Fisher English** (Cieri et al., LREC 2004 [`cieri2004fisher`]). About 2,000 h of telephone conversations,
transcribed in LDC's faster "quick transcription" style. There's more of it than Switchboard, but the labels are
a little noisier. Sold by LDC.

**ESB** (Gandhi et al., arXiv 2022 [`gandhi2022esb`]). Not a training corpus: a multi-domain ASR benchmark that
combines LibriSpeech, Common Voice, VoxPopuli, TED-LIUM, GigaSpeech, SPGISpeech, Earnings-22 and AMI. `benchmark.py`
evaluates on its test sets (`data/esb/`).

## 4. Audio understanding: ranking

The same tiers apply, now to class labels, captions and QA answers. Benchmarks are listed separately because
they're for evaluation only.

| tier | dataset | task | size | how the labels were made | access | notes |
|---|---|---|---|---|---|---|
| A | FSD50K | sound events (200 classes) | 51,197 clips, ~108 h | Human labelling against the AudioSet ontology, then refined | open (Freesound licences) | Optional (~23 GB); best open sound-event labels |
| A | ESC-50 | environmental sounds (50 classes) | 2,000 × 5 s clips | Picked and labelled by hand | open (CC BY-NC) | **On disk.** 5-fold cross-validation |
| A | UrbanSound8K | urban sounds (10 classes) | 8,732 clips ≤ 4 s | Annotated by hand | open (CC BY-NC) | 10 folds, fixed in advance |
| A | Clotho | audio captioning | ~5–7k clips of 15–30 s, 5 captions each | Crowd captions, then crowd editing and rating | open | Freesound audio, no dead links |
| A | MusicCaps | music captioning | 5,521 × 10 s | Captions and aspect lists by professional musicians | YouTube IDs | Some videos no longer online |
| A | IEMOCAP | emotion (acted dyadic dialogue) | ~12 h, 10 actors | Several annotators per turn; categorical + dimensional labels | request form | Evaluate leave-one-session-out |
| A | CREMA-D | emotion (acted) | 7,442 clips, 91 actors | Many crowd ratings per clip | open (ODbL) | **On disk.** Same 12 sentences throughout |
| A | VocalSound | non-speech vocal sounds | ~21k recordings, 3,365 people | Correct by construction: people recorded the requested sound | open | **On disk.** Laugh, sigh, cough, throat-clear, sneeze, sniff |
| A | Fluent Speech Commands | spoken intent | 30,043 utterances, 97 speakers | Correct by construction: prompted commands (action/object/location) | open | Speaker-disjoint splits |
| A | NSynth | instrument and pitch | 305,979 four-second notes | Taken from sample-library metadata | open | Synthetic or sampled notes |
| B | AudioCaps | audio captioning | ~46k AudioSet clips | Crowd captions (**1 per training clip**, 5 per val/test clip) | YouTube IDs | Standard for captioning; some audio gone |
| B | Clotho-AQA | audio QA | 1,991 clips, 6 questions each | Crowd-written questions, each answered by 3 people | open | **On disk.** Yes/no and one-word answers |
| B | Speech Commands v2 | keyword spotting | 105,829 one-second clips, 35 words | Prompted words, partly checked by hand | open | **On disk** |
| B | SLURP | spoken language understanding (intent + entities) | ~72k recordings, ~58 h | Crowd-written text, re-recorded; intents and entities annotated | open | Headset and far-field audio |
| B | MELD | emotion in conversation | ~13k utterances (*Friends*) | Several annotators, majority vote | open | TV audio with laugh tracks |
| B | VoxCeleb1 / 2 | speaker identity | 1,251 / 6,112 speakers | Automatic audio-visual pipeline (face matching + active-speaker detection) | request form | Very precise IDs, but not hand-checked |
| B | MusicNet | note transcription | 330 recordings, ~34 h | Musical scores aligned automatically, then checked | open | Small alignment error rate reported |
| B | GTZAN | music genre | 1,000 × 30 s | One curator | open mirrors | Known duplicates and mislabels: use a fault-filtered split |
| C | AudioSet | sound events (527 classes) | ~2.1M × 10 s | Candidate labels confirmed by raters, but only at clip level; accuracy varies by class | YouTube IDs | Weak labels; many dead links |
| C | VGGSound | sound events (309 classes) | ~200k × 10 s | Automatic pipeline (YouTube search + image and audio classifiers) | open mirror | Optional (~338 GB) |
| C | WavCaps | audio captioning | ~400k clips, ~7.5k h | Web descriptions rewritten by ChatGPT | open | Optional (~820 GB) |
| C | OpenAQA-5M (LTU) | audio QA / instruction tuning | ~5.6M QA pairs | Generated by GPT-3.5 from existing labels and metadata | annotations only | Audio comes from AudioSet etc. |
| C | AudioSkills (Audio Flamingo 2/3) | reasoning QA | millions of QA pairs | Generated by an LLM | open (non-commercial) | Synthetic reasoning chains |

**Benchmarks (evaluation only)**

| benchmark | what it tests | labels | on disk |
|---|---|---|---|
| MMAU | multiple-choice QA on speech, sound and music: 10k items, 27 skills (`test-mini` = 1k) | Questions written and checked by humans | yes (`test-mini`) |
| AIR-Bench | 19 foundation tasks plus open-ended chat, scored by a GPT-4 judge | Built from existing labelled sets | yes (Chat) |
| Dynamic-SUPERB | following instructions on many crowd-contributed speech/audio tasks | Built from existing labelled sets | no |
| SUPERB | 10 speech tasks for frozen representations | Built from existing labelled sets | no |

## 5. Audio understanding: summaries

### Captioning and audio QA
- **AudioCaps** (Kim et al., NAACL 2019 [`kim2019audiocaps`]). About 46k 10 s AudioSet clips with human captions
  written on Mechanical Turk, with AudioSet labels given as hints. Training clips have one caption each; val/test
  clips have five. The audio has to be fetched from YouTube. It's the standard captioning benchmark, and at our
  6.25 Hz a clip is only about 63 audio tokens.
- **Clotho** (Drossos et al., ICASSP 2020 [`drossos2020clotho`]). Freesound clips of 15–30 s with five captions
  each, written from the audio only, then edited and rated by other crowd workers. Captions are 8–20 words, with
  named entities and spelling mistakes removed. The audio is redistributed, so nothing goes missing. Its labels
  are cleaner than AudioCaps'.
- **Clotho-AQA** (Lipping et al., EUSIPCO 2022 [`lipping2022clothoaqa`]). 1,991 Clotho clips, each with six
  crowd-written questions (yes/no and one-word answers), and every question answered by three people. It's a small,
  clean first test of whether the LLM actually listens rather than guessing. It is on disk.
- **WavCaps** (Mei et al., IEEE/ACM TASLP 2024 [`mei2024wavcaps`]). About 400k clips (~7.5k h) from Freesound,
  BBC Sound Effects, SoundBible and AudioSet-strong. ChatGPT rewrote the original web descriptions into captions.
  It's large and useful for pre-aligning the projector, but the captions can be over-written or invented.
- **OpenAQA-5M / LTU** (Gong et al., ICLR 2024 [`gong2024ltu`]). About 5.6M closed- and open-ended QA pairs that
  GPT-3.5 generated from existing labels, captions and metadata of AudioSet, VGGSound, FSD50K, AudioCaps, Clotho and
  others, released with the LTU model. It's good for instruction tuning, but some answers depend on metadata
  rather than what's audible.
- **AudioSkills** (Ghosh et al., 2025 [`ghosh2025af2`]; Ghosh et al., 2025 [`ghosh2025af3`]). The synthetic
  reasoning-QA collection from the Audio Flamingo 2/3 papers. It covers temporal order, counting, attributes and
  long audio, and it's generated by LLMs.

### Sound events
- **AudioSet** (Gemmeke et al., ICASSP 2017 [`gemmeke2017audioset`]). About 2.1M 10 s YouTube clips labelled with
  527 classes from a hierarchical ontology. Raters confirmed the candidate labels, but only for the clip as a whole,
  and label accuracy varies a lot by class. It's distributed as YouTube IDs, so some clips are gone.
- **VGGSound** (Chen et al., ICASSP 2020 [`chen2020vggsound`]). About 200k 10 s clips over 309 classes, collected
  by an automatic pipeline that checks the sound source is visible and audible. It's cleaner per clip than AudioSet
  but still automatic.
- **FSD50K** (Fonseca et al., IEEE/ACM TASLP 2022 [`fonseca2022fsd50k`]). 51,197 Freesound clips (~108 h)
  annotated by humans with 200 AudioSet-ontology classes. The eval set was labelled more thoroughly than the
  dev set. Clips are Creative Commons–licensed. It's the best open sound-event dataset with human labels.
- **ESC-50** (Piczak, ACM MM 2015 [`piczak2015esc`]). 2,000 hand-picked 5 s clips over 50 balanced classes with
  5 fixed folds. It's tiny and clean, a good quick zero-shot probe. It is on disk.
- **UrbanSound8K** (Salamon et al., ACM MM 2014 [`salamon2014urbansound`]). 8,732 hand-annotated clips of up to 4 s
  over 10 urban classes. It comes with 10 predefined folds; use them, or clips from the same recording leak across
  splits.

### Speech paralinguistics
- **VoxCeleb1 / VoxCeleb2** (Nagrani et al., Interspeech 2017 [`nagrani2017voxceleb`]; Chung et al., Interspeech
  2018 [`chung2018voxceleb2`]). Celebrity interview speech from YouTube: 1,251 speakers in VoxCeleb1 and 6,112 in
  VoxCeleb2, more than 1M utterances. Labels come from an automatic audio-visual pipeline (face matching plus
  active-speaker detection). It's the standard speaker-recognition data.
- **IEMOCAP** (Busso et al., Language Resources & Evaluation 2008 [`busso2008iemocap`]). About 12 h of scripted and
  improvised two-person sessions with 10 actors. Each turn was rated by several annotators, with both categorical
  and dimensional emotion labels. You need a licence request to get it. It's the reference speech-emotion benchmark.
- **MELD** (Poria et al., ACL 2019 [`poria2019meld`]). About 13k utterances from the TV show *Friends*, labelled
  with 7 emotions and with sentiment, by majority vote. It's conversational and noisy (laugh tracks, overlapping
  speech).
- **CREMA-D** (Cao et al., IEEE Trans. Affective Computing 2014 [`cao2014cremad`]). 7,442 clips of 91 actors saying
  12 fixed sentences with 6 emotions at several intensities. Each clip has many crowd ratings. Because the words
  never change, a model has to use tone of voice, not wording. It is on disk.
- **VocalSound** (Gong et al., ICASSP 2022 [`gong2022vocalsound`]). About 21k recordings of laughter, sighs, coughs,
  throat-clearing, sneezes and sniffs from 3,365 crowd-sourced participants. Labels are correct by construction
  because people recorded the sound they were asked for. It teaches the model sounds that ASR training treats as
  noise. It is on disk.

### Spoken language understanding
- **Speech Commands v2** (Warden, arXiv 2018 [`warden2018speechcommands`]). 105,829 one-second recordings of 35
  words from 2,618 speakers, licence CC BY 4.0. It's a keyword-spotting benchmark and makes a very cheap unit test
  for the projector (about 6 audio tokens per clip). It is on disk.
- **Fluent Speech Commands** (Lugosch et al., Interspeech 2019 [`lugosch2019fsc`]). 30,043 smart-home commands from
  97 speakers, labelled with action/object/location slots (31 intents) and split by speaker. It's a simple
  end-to-end test of extracting intent directly from audio.
- **SLURP** (Bastianelli et al., EMNLP 2020 [`bastianelli2020slurp`]). About 72k recordings (~58 h) of
  assistant-style requests across 18 domains, recorded with headset and far-field microphones and annotated with
  intents and entities. It's harder and more realistic than Fluent Speech Commands.

### Music
- **MusicCaps** (Agostinelli et al., arXiv 2023 [`agostinelli2023musiclm`]). 5,521 10 s music clips, each with a
  caption and an aspect list written by professional musicians, released with MusicLM. It's the standard
  music-captioning eval; the audio is distributed as YouTube IDs.
- **GTZAN** (Tzanetakis & Cook, IEEE TSAP 2002 [`tzanetakis2002gtzan`]). 1,000 30 s clips over 10 genres. It's
  widely used despite documented duplicates and mislabels, so prefer a fault-filtered split.
- **NSynth** (Engel et al., ICML 2017 [`engel2017nsynth`]). 305,979 four-second single notes from about 1,000
  instruments, labelled with pitch, velocity, instrument family and quality attributes. Test instruments don't
  appear in training. It cleanly probes pitch and timbre.
- **MusicNet** (Thickstun et al., ICLR 2017 [`thickstun2017musicnet`]). 330 freely licensed classical recordings
  (~34 h) with more than 1M note labels, made by aligning the scores to the audio and then checking. It's a hard
  test of transcribing music with many simultaneous notes.

### Benchmarks
- **MMAU** (Sakshi et al., ICLR 2025 [`sakshi2025mmau`]). 10k audio clips with human-written multiple-choice
  questions across speech, sound and music, covering 27 skills. `test-mini` (1k) is on disk.
- **AIR-Bench** (Yang et al., ACL 2024 [`yang2024airbench`]). Foundation tasks (single-skill questions) plus
  open-ended chat, judged by GPT-4. The Chat split is on disk.
- **Dynamic-SUPERB** (Huang et al., ICASSP 2024 [`huang2024dynamicsuperb`]). A community-extended benchmark for
  following instructions in zero-shot speech and audio tasks.
- **SUPERB** (Yang et al., Interspeech 2021 [`yang2021superb`]). Ten speech tasks for frozen representations:
  content, speaker, semantics and paralinguistics.

## 6. What's on disk

`download_corpora.py` (default set ≈ 58 GB, with `data/` kept ≤ 100 GB in total):

| corpus | source repo | ~GB | role |
|---|---|---|---|
| LibriSpeech `train-clean-360`, `dev-other` | `openslr/librispeech_asr` → `data/LibriSpeech` | 23.5 | ASR train / val |
| AMI (IHM) | `edinburghcstr/ami` | 15.2 | ASR train / dev / test |
| FLEURS `en_us` | `google/fleurs` | 2.4 | ASR train (small) / dev |
| EdAcc | `edinburghcstr/edacc` | 7.0 | ASR eval only (accents) |
| Speech Commands | `barto17/speech_commands` | 4.9 | keyword spotting |
| Clotho-AQA | `lmms-lab-audio/ClothoAQA` | 1.6 | audio QA |
| ESC-50 | `ashraq/esc50` | 0.8 | sound classification |
| VocalSound | `lmms-lab-audio/vocalsound` | 0.6 | vocal-sound classification |
| CREMA-D | `confit/cremad-parquet` | 0.6 | emotion |
| MMAU test-mini | `lmms-lab-audio/mmau` | 1.2 | benchmark |
| AIR-Bench Chat | `pbcong/AIR-bench` | 0.3 | benchmark |

Already present: LibriSpeech `train-clean-100`, `dev-clean`, `test-clean` and `test-other`, plus the ESB test sets
in `data/esb/`.

**Off by default** (`--only <name>`): `librispeech_other500` (31 GB), `tedlium3` (53), `voxpopuli_en` (99),
`libritts_r` (101), `earnings22` (24, already covered by the ESB test set), `fsd50k` (23), `meld` (12),
`peoples_speech_clean` (399), `mls_eng` (705), `vggsound` (338), `wavcaps` (820).

**Gated or unavailable**:
- **GigaSpeech, SPGISpeech:** accept the terms on HF, then run `hf auth login`.
- **VCTK** (HF mirror): gated; the original is on CSTR DataShare.
- **Common Voice:** now on the Mozilla Data Collective.
- **WSJ, Switchboard, Fisher, ICSI and CHiME:** LDC or organiser licences.
- **OpenAQA:** the annotations are released, but the audio isn't.

## 7. References (BibTeX)

The entries below came from Crossref/DOI metadata, or the arXiv record where there is no DOI. They were not
written from memory.

```bibtex
@inproceedings{panayotov2015librispeech,
  title={Librispeech: An ASR corpus based on public domain audio books},
  DOI={10.1109/icassp.2015.7178964},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  year={2015},
  month=Apr,
  pages={5206--5210}
}

@inproceedings{zen2019libritts,
  title={LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  DOI={10.21437/interspeech.2019-2441},
  booktitle={Interspeech 2019},
  publisher={ISCA},
  author={Zen, Heiga and Dang, Viet and Clark, Rob and Zhang, Yu and Weiss, Ron J. and Jia, Ye and Chen, Zhifeng and Wu, Yonghui},
  year={2019},
  month=Sept,
  pages={1526--1530},
}

@inproceedings{koizumi2023librittsr,
  title={LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus},
  DOI={10.21437/interspeech.2023-1584},
  booktitle={INTERSPEECH 2023},
  publisher={ISCA},
  author={Koizumi, Yuma and Zen, Heiga and Karita, Shigeki and Ding, Yifan and Yatabe, Kohei and Morioka, Nobuyuki and Bacchiani, Michiel and Zhang, Yu and Han, Wei and Bapna, Ankur},
  year={2023},
  month=Aug,
  pages={5496--5500},
}

@inproceedings{kahn2020librilight,
  title={Libri-Light: A Benchmark for ASR with Limited or No Supervision},
  DOI={10.1109/icassp40776.2020.9052942},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Kahn, J. and Riviere, M. and Zheng, W. and Kharitonov, E. and Xu, Q. and Mazare, P.E. and Karadayi, J. and Liptchinsky, V. and Collobert, R. and Fuegen, C. and Likhomanenko, T. and Synnaeve, G. and Joulin, A. and Mohamed, A. and Dupoux, E.},
  year={2020},
  month=May,
  pages={7669--7673}
}

@inproceedings{kang2024libriheavy,
  title={Libriheavy: A 50,000 Hours ASR Corpus with Punctuation Casing and Context},
  DOI={10.1109/icassp48485.2024.10447120},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Kang, Wei and Yang, Xiaoyu and Yao, Zengwei and Kuang, Fangjun and Yang, Yifan and Guo, Liyong and Lin, Long and Povey, Daniel},
  year={2024},
  month=Apr,
  pages={10991--10995}
}

@inbook{hernandez2018tedlium3,
  title={TED-LIUM 3: Twice as Much Data and Corpus Repartition for Experiments on Speaker Adaptation},
  ISBN={9783319995793},
  ISSN={1611-3349},
  DOI={10.1007/978-3-319-99579-3_21},
  booktitle={Speech and Computer},
  publisher={Springer International Publishing},
  author={Hernandez, François and Nguyen, Vincent and Ghannay, Sahar and Tomashenko, Natalia and Estève, Yannick},
  year={2018},
  pages={198--208}
}

@misc{ardila2020commonvoice,
  doi = {10.48550/ARXIV.1912.06670},
  author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M. and Weber, Gregor},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license},
  note = {Published in Proc. LREC 2020, pp. 4218--4222}
}

@inproceedings{wang2021voxpopuli,
  title={VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
  DOI={10.18653/v1/2021.acl-long.80},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  publisher={Association for Computational Linguistics},
  author={Wang, Changhan and Riviere, Morgane and Lee, Ann and Wu, Anne and Talnikar, Chaitanya and Haziza, Daniel and Williamson, Mary and Pino, Juan and Dupoux, Emmanuel},
  year={2021},
  pages={993--1003}
}

@inbook{carletta2005ami,
  title={The AMI Meeting Corpus: A Pre-announcement},
  ISBN={9783540325505},
  ISSN={1611-3349},
  DOI={10.1007/11677482_3},
  booktitle={Machine Learning for Multimodal Interaction},
  publisher={Springer Berlin Heidelberg},
  author={Carletta, Jean and Ashby, Simone and Bourban, Sebastien and Flynn, Mike and Guillemot, Mael and Hain, Thomas and Kadlec, Jaroslav and Karaiskos, Vasilis and Kraaij, Wessel and Kronenthal, Melissa and Lathoud, Guillaume and Lincoln, Mike and Lisowska, Agnes and McCowan, Iain and Post, Wilfried and Reidsma, Dennis and Wellner, Pierre},
  year={2006},
  pages={28--39}
}

@inproceedings{janin2003icsi,
  title={The ICSI Meeting Corpus},
  volume={1},
  DOI={10.1109/icassp.2003.1198793},
  booktitle={2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings. (ICASSP ’03).},
  publisher={IEEE},
  author={Janin, A. and Baron, D. and Edwards, J. and Ellis, D. and Gelbart, D. and Morgan, N. and Peskin, B. and Pfau, T. and Shriberg, E. and Stolcke, A. and Wooters, C.},
  pages={I--364--I--367},
  year={2003}
}

@inproceedings{barker2018chime5,
  title={The Fifth ‘CHiME’ Speech Separation and Recognition Challenge: Dataset, Task and Baselines},
  DOI={10.21437/interspeech.2018-1768},
  booktitle={Interspeech 2018},
  publisher={ISCA},
  author={Barker, Jon and Watanabe, Shinji and Vincent, Emmanuel and Trmal, Jan},
  year={2018},
  month=Sept,
  pages={1561--1565},
}

@inproceedings{watanabe2020chime6,
  title={CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings},
  DOI={10.21437/chime.2020-1},
  booktitle={6th International Workshop on Speech Processing in Everyday Environments (CHiME 2020)},
  publisher={ISCA},
  author={Watanabe, Shinji and Mandel, Michael and Barker, Jon and Vincent, Emmanuel and Arora, Ashish and Chang, Xuankai and Khudanpur, Sanjeev and Manohar, Vimal and Povey, Daniel and Raj, Desh and Snyder, David and Subramanian, Aswin Shanmugam and Trmal, Jan and Yair, Bar Ben and Boeddeker, Christoph and Ni, Zhaoheng and Fujita, Yusuke and Horiguchi, Shota and Kanda, Naoyuki and Yoshioka, Takuya and Ryant, Neville},
  year={2020},
  month=May,
  pages={1--7},
}

@inproceedings{chen2021gigaspeech,
  title={GigaSpeech: An Evolving, Multi-Domain ASR Corpus with 10,000 Hours of Transcribed Audio},
  DOI={10.21437/interspeech.2021-1965},
  booktitle={Interspeech 2021},
  publisher={ISCA},
  author={Chen, Guoguo and Chai, Shuzhou and Wang, Guan-Bo and Du, Jiayu and Zhang, Wei-Qiang and Weng, Chao and Su, Dan and Povey, Daniel and Trmal, Jan and Zhang, Junbo and Jin, Mingjie and Khudanpur, Sanjeev and Watanabe, Shinji and Zhao, Shuaijiang and Zou, Wei and Li, Xiangang and Yao, Xuchen and Wang, Yongqing and You, Zhao and Yan, Zhiyong},
  year={2021},
  month=Aug,
  pages={3670--3674},
}

@inproceedings{oneill2021spgispeech,
  title={SPGISpeech: 5,000 Hours of Transcribed Financial Audio for Fully Formatted End-to-End Speech Recognition},
  DOI={10.21437/interspeech.2021-1860},
  booktitle={Interspeech 2021},
  publisher={ISCA},
  author={O’Neill, Patrick K. and Lavrukhin, Vitaly and Majumdar, Somshubra and Noroozi, Vahid and Zhang, Yuekai and Kuchaiev, Oleksii and Balam, Jagadeesh and Dovzhenko, Yuliya and Freyberg, Keenan and Shulman, Michael D. and Ginsburg, Boris and Watanabe, Shinji and Kucsko, Georg},
  year={2021},
  month=Aug,
  pages={1434--1438},
}

@inproceedings{delrio2021earnings21,
  title={Earnings-21: A Practical Benchmark for ASR in the Wild},
  DOI={10.21437/interspeech.2021-1915},
  booktitle={Interspeech 2021},
  publisher={ISCA},
  author={Del Rio, Miguel and Delworth, Natalie and Westerman, Ryan and Huang, Michelle and Bhandari, Nishchal and Palakapilly, Joseph and McNamara, Quinten and Dong, Joshua and Żelasko, Piotr and Jetté, Miguel},
  year={2021},
  month=Aug,
  pages={3465--3469},
}

@misc{delrio2022earnings22,
  doi = {10.48550/ARXIV.2203.15591},
  author = {Del Rio, Miguel and Ha, Peter and McNamara, Quinten and Miller, Corey and Chandra, Shipra},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Earnings-22: A Practical Benchmark for Accents in the Wild},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}

@misc{galvez2021peoplesspeech,
  doi = {10.48550/ARXIV.2111.09344},
  author = {Galvez, Daniel and Diamos, Greg and Ciro, Juan and Cerón, Juan Felipe and Achorn, Keith and Gopi, Anjali and Kanter, David and Lam, Maximilian and Mazumder, Mark and Reddi, Vijay Janapa},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {The People's Speech: A Large-Scale Diverse English Speech Recognition Dataset for Commercial Usage},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license},
  note = {NeurIPS 2021 Datasets and Benchmarks Track}
}

@inproceedings{pratap2020mls,
  title={MLS: A Large-Scale Multilingual Dataset for Speech Research},
  DOI={10.21437/interspeech.2020-2826},
  booktitle={Interspeech 2020},
  publisher={ISCA},
  author={Pratap, Vineel and Xu, Qiantong and Sriram, Anuroop and Synnaeve, Gabriel and Collobert, Ronan},
  year={2020},
  month=Oct,
  pages={2757--2761},
}

@inproceedings{conneau2023fleurs,
  title={FLEURS: FEW-Shot Learning Evaluation of Universal Representations of Speech},
  DOI={10.1109/slt54892.2023.10023141},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
  publisher={IEEE},
  author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  year={2023},
  month=Jan,
  pages={798--805}
}

@inproceedings{li2023yodas,
  title={Yodas: Youtube-Oriented Dataset for Audio and Speech},
  DOI={10.1109/asru57964.2023.10389689},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  publisher={IEEE},
  author={Li, Xinjian and Takamichi, Shinnosuke and Saeki, Takaaki and Chen, William and Shiota, Sayaka and Watanabe, Shinji},
  year={2023},
  month=Dec,
  pages={1--8}
}

@inproceedings{he2024emilia,
  title={Emilia: An Extensive, Multilingual, and Diverse Speech Dataset For Large-Scale Speech Generation},
  DOI={10.1109/slt61566.2024.10832365},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)},
  publisher={IEEE},
  author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Wang, Yuancheng and Chen, Kai and Zhang, Pengyuan and Wu, Zhizheng},
  year={2024},
  month=Dec,
  pages={885--890}
}

@inproceedings{koluguri2025granary,
  title={Granary: Speech Recognition and Translation Dataset in 25 European Languages},
  DOI={10.21437/interspeech.2025-540},
  booktitle={Interspeech 2025},
  publisher={ISCA},
  author={Rao Koluguri, Nithin and Sekoyan, Monica and Zelenfroynd, George and Meister, Sasha and Ding, Shuoyang and Kostandian, Sofia and Huang, He and Karpov#, Nikolay and Balam, Jagadeesh and Lavrukhin, Vitaly and Peng, Yifan and Papi, Sara and Gaido, Marco and Brutti, Alessio and Ginsburg, Boris},
  year={2025},
  month=Aug,
  pages={3923--3927},
}

@inproceedings{sanabria2023edacc,
  title={The Edinburgh International Accents of English Corpus: Towards the Democratization of English ASR},
  DOI={10.1109/icassp49357.2023.10095057},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Sanabria, Ramon and Bogoychev, Nikolay and Markl, Nina and Carmantini, Andrea and Klejch, Ondrej and Bell, Peter},
  year={2023},
  month=June,
  pages={1--5}
}

@misc{yamagishi2019vctk,
  doi = {10.7488/DS/2645},
  author = {Yamagishi, Junichi and Veaux, Christophe and MacDonald, Kirsten},
  keywords = {Speech synthesis, Deep learning, Machine learning, Speaker recognition, Mathematical and Computer Sciences},
  language = {en},
  title = {CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)},
  publisher = {University of Edinburgh. The Centre for Speech Technology Research (CSTR)},
  year = {2019},
  copyright = {Creative Commons Attribution 4.0 International Public License}
}

@inproceedings{paul1992wsj,
  title={The design for the wall street journal-based CSR corpus},
  DOI={10.21437/icslp.1992-277},
  booktitle={2nd International Conference on Spoken Language Processing (ICSLP 1992)},
  publisher={ISCA},
  author={Paul, Douglas B. and Baker, Janet M.},
  year={1992},
  month=Oct,
  pages={899--902},
}

@inproceedings{godfrey1992switchboard,
  title={SWITCHBOARD: telephone speech corpus for research and development},
  DOI={10.1109/icassp.1992.225858},
  booktitle={[Proceedings] ICASSP-92: 1992 IEEE International Conference on Acoustics, Speech, and Signal Processing},
  publisher={IEEE},
  author={Godfrey, J.J. and Holliman, E.C. and McDaniel, J.},
  year={1992},
  pages={517--520 vol.1}
}

@misc{gandhi2022esb,
  doi = {10.48550/ARXIV.2210.13352},
  author = {Gandhi, Sanchit and von Platen, Patrick and Rush, Alexander M.},
  keywords = {Computation and Language (cs.CL), Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {ESB: A Benchmark For Multi-Domain End-to-End Speech Recognition},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@inproceedings{gemmeke2017audioset,
  title={Audio Set: An ontology and human-labeled dataset for audio events},
  DOI={10.1109/icassp.2017.7952261},
  booktitle={2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Gemmeke, Jort F. and Ellis, Daniel P. W. and Freedman, Dylan and Jansen, Aren and Lawrence, Wade and Moore, R. Channing and Plakal, Manoj and Ritter, Marvin},
  year={2017},
  month=Mar,
  pages={776--780}
}

@inproceedings{kim2019audiocaps,
  title={AudioCaps: Generating Captions for Audios in The Wild},
  DOI={10.18653/v1/n19-1011},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  publisher={Association for Computational Linguistics},
  author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  year={2019},
  pages={119--132}
}

@inproceedings{drossos2020clotho,
  title={Clotho: an Audio Captioning Dataset},
  DOI={10.1109/icassp40776.2020.9052990},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas},
  year={2020},
  month=May,
  pages={736--740}
}

@inproceedings{lipping2022clothoaqa,
  title={Clotho-AQA: A Crowdsourced Dataset for Audio Question Answering},
  DOI={10.23919/eusipco55093.2022.9909680},
  booktitle={2022 30th European Signal Processing Conference (EUSIPCO)},
  publisher={IEEE},
  author={Lipping, Samuel and Sudarsanam, Parthasaarathy and Drossos, Konstantinos and Virtanen, Tuomas},
  year={2022},
  month=Aug,
  pages={1140--1144}
}

@article{mei2024wavcaps,
  title={WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research},
  volume={32},
  ISSN={2329-9304},
  DOI={10.1109/taslp.2024.3419446},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  publisher={Institute of Electrical and Electronics Engineers (IEEE)},
  author={Mei, Xinhao and Meng, Chutong and Liu, Haohe and Kong, Qiuqiang and Ko, Tom and Zhao, Chengqi and Plumbley, Mark D. and Zou, Yuexian and Wang, Wenwu},
  year={2024},
  pages={3339--3354}
}

@misc{gong2024ltu,
  doi = {10.48550/ARXIV.2305.10790},
  author = {Gong, Yuan and Luo, Hongyin and Liu, Alexander H. and Karlinsky, Leonid and Glass, James},
  keywords = {Audio and Speech Processing (eess.AS), Sound (cs.SD), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Listen, Think, and Understand},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International},
  note = {Published at ICLR 2024}
}

@misc{ghosh2025af2,
  doi = {10.48550/ARXIV.2503.03983},
  author = {Ghosh, Sreyan and Kong, Zhifeng and Kumar, Sonal and Sakshi, S and Kim, Jaehyeon and Ping, Wei and Valle, Rafael and Manocha, Dinesh and Catanzaro, Bryan},
  keywords = {Sound (cs.SD), Computation and Language (cs.CL), Machine Learning (cs.LG), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International},
  note = {Published at ICML 2025}
}

@inproceedings{ghosh2025af3,
  title={Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models},
  DOI={10.52202/085713-1396},
  booktitle={Advances in Neural Information Processing Systems 38},
  publisher={Neural Information Processing Systems Foundation, Inc. (NeurIPS)},
  author={Ghosh, Sreyan and Goel, Arushi and Kim, Jaehyeon and Kumar, Sonal and Kong, Zhifeng and Lee, Sang-gil and Yang, Chao-Han and Duraiswami, Ramani and Manocha, Dinesh and Valle, Rafael and Catanzaro, Bryan},
  year={2025},
  pages={46789--46856},
}

@inproceedings{chen2020vggsound,
  title={Vggsound: A Large-Scale Audio-Visual Dataset},
  DOI={10.1109/icassp40776.2020.9053174},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  year={2020},
  month=May,
  pages={721--725}
}

@article{fonseca2022fsd50k,
  title={FSD50K: An Open Dataset of Human-Labeled Sound Events},
  volume={30},
  ISSN={2329-9304},
  DOI={10.1109/taslp.2021.3133208},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  publisher={Institute of Electrical and Electronics Engineers (IEEE)},
  author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  year={2022},
  pages={829--852}
}

@inproceedings{piczak2015esc,
  title={ESC: Dataset for Environmental Sound Classification},
  DOI={10.1145/2733373.2806390},
  booktitle={Proceedings of the 23rd ACM international conference on Multimedia},
  publisher={ACM},
  author={Piczak, Karol J.},
  year={2015},
  month=Oct,
  pages={1015--1018},
}

@inproceedings{salamon2014urbansound,
  title={A Dataset and Taxonomy for Urban Sound Research},
  DOI={10.1145/2647868.2655045},
  booktitle={Proceedings of the 22nd ACM international conference on Multimedia},
  publisher={ACM},
  author={Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
  year={2014},
  month=Nov,
  pages={1041--1044},
}

@inproceedings{nagrani2017voxceleb,
  title={VoxCeleb: A Large-Scale Speaker Identification Dataset},
  DOI={10.21437/interspeech.2017-950},
  booktitle={Interspeech 2017},
  publisher={ISCA},
  author={Nagrani, Arsha and Chung, Joon Son and Zisserman, Andrew},
  year={2017},
  month=Aug,
  pages={2616--2620},
}

@inproceedings{chung2018voxceleb2,
  title={VoxCeleb2: Deep Speaker Recognition},
  DOI={10.21437/interspeech.2018-1929},
  booktitle={Interspeech 2018},
  publisher={ISCA},
  author={Chung, Joon Son and Nagrani, Arsha and Zisserman, Andrew},
  year={2018},
  month=Sept,
  pages={1086--1090},
}

@article{busso2008iemocap,
  title={IEMOCAP: interactive emotional dyadic motion capture database},
  volume={42},
  ISSN={1574-0218},
  DOI={10.1007/s10579-008-9076-6},
  number={4},
  journal={Language Resources and Evaluation},
  publisher={Springer Science and Business Media LLC},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N. and Lee, Sungbok and Narayanan, Shrikanth S.},
  year={2008},
  month=Nov,
  pages={335--359}
}

@inproceedings{poria2019meld,
  title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  DOI={10.18653/v1/p19-1050},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  publisher={Association for Computational Linguistics},
  author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  year={2019},
  pages={527--536}
}

@article{cao2014cremad,
  title={CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset},
  volume={5},
  ISSN={1949-3045},
  DOI={10.1109/taffc.2014.2336244},
  number={4},
  journal={IEEE Transactions on Affective Computing},
  publisher={Institute of Electrical and Electronics Engineers (IEEE)},
  author={Cao, Houwei and Cooper, David G. and Keutmann, Michael K. and Gur, Ruben C. and Nenkova, Ani and Verma, Ragini},
  year={2014},
  month=Oct,
  pages={377--390}
}

@inproceedings{gong2022vocalsound,
  title={Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
  DOI={10.1109/icassp43922.2022.9746828},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Gong, Yuan and Yu, Jin and Glass, James},
  year={2022},
  month=May,
  pages={151--155}
}

@misc{warden2018speechcommands,
  doi = {10.48550/ARXIV.1804.03209},
  author = {Warden, Pete},
  keywords = {Computation and Language (cs.CL), Human-Computer Interaction (cs.HC), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
  publisher = {arXiv},
  year = {2018},
  copyright = {Creative Commons Attribution 4.0 International}
}

@inproceedings{lugosch2019fsc,
  title={Speech Model Pre-Training for End-to-End Spoken Language Understanding},
  DOI={10.21437/interspeech.2019-2396},
  booktitle={Interspeech 2019},
  publisher={ISCA},
  author={Lugosch, Loren and Ravanelli, Mirco and Ignoto, Patrick and Tomar, Vikrant Singh and Bengio, Yoshua},
  year={2019},
  month=Sept,
  pages={814--818},
}

@inproceedings{bastianelli2020slurp,
  title={SLURP: A Spoken Language Understanding Resource Package},
  DOI={10.18653/v1/2020.emnlp-main.588},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  publisher={Association for Computational Linguistics},
  author={Bastianelli, Emanuele and Vanzo, Andrea and Swietojanski, Pawel and Rieser, Verena},
  year={2020},
  pages={7252--7262}
}

@misc{agostinelli2023musiclm,
  doi = {10.48550/ARXIV.2301.11325},
  author = {Agostinelli, Andrea and Denk, Timo I. and Borsos, Zalán and Engel, Jesse and Verzetti, Mauro and Caillon, Antoine and Huang, Qingqing and Jansen, Aren and Roberts, Adam and Tagliasacchi, Marco and Sharifi, Matt and Zeghidour, Neil and Frank, Christian},
  keywords = {Sound (cs.SD), Machine Learning (cs.LG), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {MusicLM: Generating Music From Text},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license},
  note = {MusicCaps dataset introduced here}
}

@article{tzanetakis2002gtzan,
  title={Musical genre classification of audio signals},
  volume={10},
  ISSN={1558-2353},
  DOI={10.1109/tsa.2002.800560},
  number={5},
  journal={IEEE Transactions on Speech and Audio Processing},
  publisher={Institute of Electrical and Electronics Engineers (IEEE)},
  author={Tzanetakis, G. and Cook, P.},
  year={2002},
  month=July,
  pages={293--302}
}

@misc{engel2017nsynth,
  doi = {10.48550/ARXIV.1704.01279},
  author = {Engel, Jesse and Resnick, Cinjon and Roberts, Adam and Dieleman, Sander and Eck, Douglas and Simonyan, Karen and Norouzi, Mohammad},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Sound (cs.SD), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders},
  publisher = {arXiv},
  year = {2017},
  copyright = {arXiv.org perpetual, non-exclusive license},
  note = {Published at ICML 2017}
}

@misc{thickstun2017musicnet,
  doi = {10.48550/ARXIV.1611.09827},
  author = {Thickstun, John and Harchaoui, Zaid and Kakade, Sham},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), Sound (cs.SD), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learning Features of Music from Scratch},
  publisher = {arXiv},
  year = {2016},
  copyright = {arXiv.org perpetual, non-exclusive license},
  note = {Published at ICLR 2017}
}

@inproceedings{yang2024airbench,
  title={AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension},
  DOI={10.18653/v1/2024.acl-long.109},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  publisher={Association for Computational Linguistics},
  author={Yang, Qian and Xu, Jin and Liu, Wenrui and Chu, Yunfei and Jiang, Ziyue and Zhou, Xiaohuan and Leng, Yichong and Lv, Yuanjun and Zhao, Zhou and Zhou, Chang and Zhou, Jingren},
  year={2024},
  pages={1979--1998}
}

@inproceedings{huang2024dynamicsuperb,
  title={Dynamic-Superb: Towards a Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark For Speech},
  DOI={10.1109/icassp48485.2024.10448257},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  publisher={IEEE},
  author={Huang, Chien-Yu and Lu, Ke-Han and Wang, Shih-Heng and Hsiao, Chi-Yuan and Kuan, Chun-Yi and Wu, Haibin and Arora, Siddhant and Chang, Kai-Wei and Shi, Jiatong and Peng, Yifan and Sharma, Roshan and Watanabe, Shinji and Ramakrishnan, Bhiksha and Shehata, Shady and Lee, Hung-Yi},
  year={2024},
  month=Apr,
  pages={12136--12140}
}

@inproceedings{yang2021superb,
  title={SUPERB: Speech Processing Universal PERformance Benchmark},
  DOI={10.21437/interspeech.2021-1775},
  booktitle={Interspeech 2021},
  publisher={ISCA},
  author={Yang, Shu-wen and Chi, Po-Han and Chuang, Yung-Sung and Lai, Cheng-I Jeff and Lakhotia, Kushal and Lin, Yist Y. and Liu, Andy T. and Shi, Jiatong and Chang, Xuankai and Lin, Guan-Ting and Huang, Tzu-Hsien and Tseng, Wei-Cheng and Lee, Ko-tik and Liu, Da-Rong and Huang, Zili and Dong, Shuyan and Li, Shang-Wen and Watanabe, Shinji and Mohamed, Abdelrahman and Lee, Hung-yi},
  year={2021},
  month=Aug,
  pages={1194--1198},
}

@misc{sakshi2025mmau,
  doi = {10.48550/ARXIV.2410.19168},
  author = {Sakshi, S and Tyagi, Utkarsh and Kumar, Sonal and Seth, Ashish and Selvakumar, Ramaneswaran and Nieto, Oriol and Duraiswami, Ramani and Ghosh, Sreyan and Manocha, Dinesh},
  keywords = {Audio and Speech Processing (eess.AS), Artificial Intelligence (cs.AI), Computation and Language (cs.CL), Sound (cs.SD), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark},
  publisher = {arXiv},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International},
  note = {Published at ICLR 2025}
}

@inproceedings{cieri2004fisher,
  title = {The {Fisher} Corpus: a Resource for the Next Generations of Speech-to-Text},
  author = {Cieri, Christopher and Miller, David and Walker, Kevin},
  booktitle = {Proceedings of the Fourth International Conference on Language Resources and Evaluation (LREC'04)},
  pages = {69--71},
  year = {2004},
  publisher = {European Language Resources Association (ELRA)}
}
```
