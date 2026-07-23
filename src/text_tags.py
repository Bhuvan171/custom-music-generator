"""
Free text -> MTG-Jamendo tags, via sentence embeddings.

The DiT has no text encoder: it conditions on a fixed 195-word tag vocabulary. This module is
a RETRIEVAL layer in front of it — "upbeat jazzy piano" becomes
genre---jazz, instrument---piano, mood/theme---upbeat. Nothing is retrained, and no information
is created: the model still receives exactly the ~30-40 bits those tags carry. This buys a usable
interface, not better audio.

Two details that matter for match quality:

1. TAG NAMES ARE NOT ENGLISH. They are lowercase and concatenated: "electricguitar",
   "drumnbass", "rocknroll", "singersongwriter". Embedded raw, a sentence encoder sees garbage
   subword soup and similarity becomes noise. PHRASES below expands them, and adds category
   context ("rhodes" -> "Rhodes electric piano") where the bare token is ambiguous.

2. WHOLE-PROMPT SIMILARITY IS NOT ENOUGH. "a fast energetic rock song with electric guitar"
   embedded as one vector is not especially close to "piano" OR "guitar" — averaging over the
   sentence dilutes every specific concept. So each tag is scored against the full prompt AND
   against every word and bigram in it, taking the max. That way a single word like "piano"
   can pull in its tag even inside a long sentence.
"""

import json
import os

import numpy as np

import config

MODEL_NAME = "all-MiniLM-L6-v2"       # 80 MB, fast, good enough for 195 short phrases
CACHE_PATH = "data/tag_embeddings.npz"

# Only tags whose bare token is not plain English. Everything else is used as-is.
PHRASES = {
    # --- genre ---
    "60s": "1960s music", "70s": "1970s music", "80s": "1980s music", "90s": "1990s music",
    "acidjazz": "acid jazz", "alternativerock": "alternative rock", "bossanova": "bossa nova",
    "bluesrock": "blues rock", "classicrock": "classic rock", "darkambient": "dark ambient",
    "deephouse": "deep house", "drumnbass": "drum and bass", "easylistening": "easy listening",
    "edm": "electronic dance music", "electropop": "electro pop", "ethnicrock": "ethnic rock",
    "hard": "hard aggressive music", "hardrock": "hard rock", "heavymetal": "heavy metal",
    "hiphop": "hip hop", "idm": "intelligent dance music",
    "instrumentalpop": "instrumental pop", "instrumentalrock": "instrumental rock",
    "jazzfunk": "jazz funk", "jazzfusion": "jazz fusion", "newage": "new age",
    "newwave": "new wave", "popfolk": "pop folk", "poprock": "pop rock",
    "postrock": "post rock", "punkrock": "punk rock", "rnb": "rhythm and blues",
    "rocknroll": "rock and roll", "singersongwriter": "singer songwriter",
    "synthpop": "synth pop", "triphop": "trip hop", "worldfusion": "world fusion",
    "chanson": "french chanson", "ethno": "ethnic music", "club": "club dance music",
    # --- instrument ---
    "acousticbassguitar": "acoustic bass guitar", "acousticguitar": "acoustic guitar",
    "classicalguitar": "classical guitar", "doublebass": "double bass",
    "drummachine": "drum machine", "electricguitar": "electric guitar",
    "electricpiano": "electric piano", "pipeorgan": "pipe organ",
    "rhodes": "Rhodes electric piano", "pad": "synthesizer pad",
    "beat": "drum beat", "computer": "computer generated sounds",
    "sampler": "sampler", "voice": "vocals singing voice", "brass": "brass section",
    "strings": "string section", "horn": "french horn",
    # --- mood/theme ---
    "ambiental": "ambient atmosphere", "film": "film soundtrack", "movie": "movie soundtrack",
    "game": "video game music", "space": "spacey cosmic", "nature": "nature outdoors",
    "sport": "sports energetic", "trailer": "movie trailer", "deep": "deep profound",
    "background": "background music", "corporate": "corporate business",
    "commercial": "commercial advertising", "advertising": "advertising jingle",
    "documentary": "documentary film", "children": "children kids",
}


def tag_to_text(tag):
    """'genre---electricguitar' -> 'electric guitar' (plus category context for the encoder)."""
    cat, name = tag.split("---")
    phrase = PHRASES.get(name, name)
    if cat == "instrument":
        return f"{phrase} instrument"
    if cat == "mood/theme":
        return f"{phrase} mood"
    return f"{phrase} music"


# Fragments that must never be queried ALONE. tag_to_text() appends "music"/"instrument"/"mood"
# to every tag, so the bare word "music" scored 0.806 against "rock music" — enough to inject
# genre---rock into "epic orchestral movie trailer music". Generic words carry no musical
# intent on their own; they are still useful inside a bigram ("dance music", "drum machine"),
# so they are only dropped as standalone unigrams.
_STOP = {"music", "song", "track", "sound", "sounds", "tune", "with", "and", "a", "an", "the",
         "for", "of", "in", "on", "some", "that", "this", "very", "like", "make", "made",
         "style", "vibe", "vibes", "feel", "feeling", "mood", "instrument", "genre"}


def _ngrams(text, n_max=3):
    """The prompt itself, plus every 1..n_max word window — see the module docstring."""
    words = [w for w in text.lower().replace(",", " ").replace("/", " ").split() if w]
    out = [text]
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            frag = words[i:i + n]
            if n == 1 and frag[0] in _STOP:
                continue
            out.append(" ".join(frag))
    return list(dict.fromkeys(out))       # dedupe, keep order


class TextTagMatcher:
    def __init__(self, vocab_path=None, verbose=True):
        from sentence_transformers import SentenceTransformer
        with open(vocab_path or config.VOCAB_PATH) as f:
            self.vocab = json.load(f)
        self.tags = sorted(self.vocab)
        self.model = SentenceTransformer(MODEL_NAME)

        # Tag embeddings never change; cache them, keyed on the vocabulary itself so an edit
        # to PHRASES or the vocab invalidates the cache instead of silently using stale vectors.
        # hashlib, NOT hash(): Python randomizes string hashing per process (PYTHONHASHSEED), so
        # a built-in hash produces a different key every run and the cache never hits.
        import hashlib
        key = hashlib.sha1(
            (repr(self.tags) + repr(sorted(PHRASES.items())) + MODEL_NAME).encode()
        ).hexdigest()
        emb = None
        if os.path.exists(CACHE_PATH):
            d = np.load(CACHE_PATH, allow_pickle=True)
            if str(d.get("key", "")) == key:
                emb = d["emb"]
        if emb is None:
            if verbose:
                print(f"embedding {len(self.tags)} tags with {MODEL_NAME} (one-off)...")
            emb = self.model.encode([tag_to_text(t) for t in self.tags],
                                    normalize_embeddings=True, show_progress_bar=False)
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            np.savez(CACHE_PATH, emb=emb, key=key)
        self.emb = emb

    def match(self, text, per_category=(2, 3, 2), threshold=0.42):
        """
        text -> [(tag, score), ...]

        per_category: how many genre / instrument / mood tags to take at most. Selecting per
        category rather than a global top-k stops a prompt like "jazz piano" returning five
        near-synonymous genres and no instrument.
        threshold: minimum cosine similarity. Without it, an unrelated prompt still returns its
        195 least-bad matches with total confidence.
        """
        q = self.model.encode(_ngrams(text), normalize_embeddings=True, show_progress_bar=False)
        sims = (self.emb @ q.T).max(axis=1)          # best-matching fragment per tag

        caps = dict(zip(("genre", "instrument", "mood/theme"), per_category))
        chosen, used = [], {k: 0 for k in caps}
        for i in np.argsort(-sims):
            tag, s = self.tags[i], float(sims[i])
            if s < threshold:
                break
            cat = tag.split("---")[0]
            if used[cat] < caps[cat]:
                chosen.append((tag, s))
                used[cat] += 1
            if all(used[c] >= caps[c] for c in caps):
                break
        return chosen

    def indices(self, text, **kw):
        return [self.vocab[t] for t, _ in self.match(text, **kw)]
