from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import functools, re, logging, os

# JPype for Zemberek integration
from jpype import startJVM, getDefaultJVMPath, JClass, isJVMStarted

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("zemberek-svc")

app = FastAPI(title="TR Lemmatizer (Zemberek)")

# Zemberek integration
_zemberek_morphology = None


def init_zemberek():
    """Initialize Zemberek"""
    global _zemberek_morphology

    try:
        # Check if zemberek-full.jar exists
        zemberek_jar = os.path.join(os.getcwd(), "zemberek-full.jar")
        if not os.path.exists(zemberek_jar):
            raise FileNotFoundError(f"Zemberek JAR not found at {zemberek_jar}")

        # Start JVM if not already started
        if not isJVMStarted():
            # Simple JVM startup with just Zemberek jar
            startJVM(getDefaultJVMPath(), f"-Djava.class.path={zemberek_jar}")

        # Initialize Zemberek morphology
        TurkishMorphology = JClass("zemberek.morphology.TurkishMorphology")
        _zemberek_morphology = TurkishMorphology.createWithDefaults()
        log.info("Zemberek initialized successfully")
        return True
    except Exception as e:
        log.error(f"Failed to initialize Zemberek: {e}")
        raise


def get_zemberek_morphology():
    """Get Zemberek morphology instance"""
    return _zemberek_morphology


def zemberek_lemmatize_text(text: str) -> str:
    """Lemmatize single text using Zemberek with legal mode (preserve adjectives)"""
    if not _zemberek_morphology:
        raise ValueError("Zemberek not initialized")

    try:
        # Use analyzeAndDisambiguate for better context awareness
        analysis = _zemberek_morphology.analyzeAndDisambiguate(text)
        best_analysis = analysis.bestAnalysis()

        # Extract lemmas with legal mode (preserve adjectives)
        lemmas = []
        for i in range(best_analysis.size()):
            analysis_item = best_analysis.get(i)
            lemma_list = analysis_item.getLemmas()

            if lemma_list and lemma_list.size() > 0:
                # Get the best lemma based on context - prefer longer, more meaningful lemmas
                if lemma_list.size() == 1:
                    best_lemma = str(lemma_list.get(0))
                else:
                    # Multiple lemmas available - choose the most appropriate one
                    lemma_candidates = [
                        str(lemma_list.get(j)) for j in range(lemma_list.size())
                    ]
                    best_lemma = _select_best_lemma(
                        lemma_candidates, str(analysis_item.surfaceForm())
                    )
                surface_form = str(analysis_item.surfaceForm())
                pos = (
                    str(analysis_item.getPos())
                    if hasattr(analysis_item, "getPos")
                    else ""
                )

                # Handle UNK tokens - try to normalize them, otherwise keep original
                if best_lemma == "UNK" or best_lemma == "Unknown":
                    normalized_unk = _normalize_unk_word(surface_form)
                    lemmas.append(normalized_unk)  # Use normalized or original
                # Check for meaningful passive forms that should be preserved
                elif _is_meaningful_passive_form(
                    surface_form,
                    [str(lemma_list.get(j)) for j in range(lemma_list.size())],
                ):
                    lemmas.append(surface_form)  # Keep meaningful passive forms
                # Check for incorrect lemmatization corrections
                elif _should_correct_lemma(surface_form, best_lemma):
                    corrected_lemma = _get_corrected_lemma(surface_form, best_lemma)
                    lemmas.append(corrected_lemma)  # Use corrected lemma
                # Normalize institutional terms to base form, preserve others
                elif _is_institutional_term(surface_form, best_lemma):
                    normalized_form = _normalize_institutional_term(
                        surface_form, best_lemma
                    )
                    lemmas.append(normalized_form)  # Use normalized form
                # Legal mode: preserve adjectives and proper nouns
                elif _should_preserve_legal(surface_form, best_lemma, pos):
                    lemmas.append(surface_form)  # Keep original form
                else:
                    lemmas.append(best_lemma)  # Use lemmatized form
            else:
                # Fallback to the original token
                lemmas.append(str(analysis_item.surfaceForm()))

        return " ".join(lemmas)
    except Exception as e:
        log.error(f"Zemberek lemmatization failed: {e}")
        raise


def _select_best_lemma(lemma_candidates: List[str], surface_form: str) -> str:
    """Select the best lemma from multiple candidates"""
    if not lemma_candidates:
        return surface_form

    if len(lemma_candidates) == 1:
        return lemma_candidates[0]

    # Strategy 1: Prefer longer lemmas (usually more meaningful)
    # Strategy 2: Avoid very short lemmas that might be stems
    # Strategy 3: Prefer lemmas that are closer to the surface form

    # Filter out very short lemmas (likely stems)
    meaningful_lemmas = [lemma for lemma in lemma_candidates if len(lemma) >= 3]
    if not meaningful_lemmas:
        meaningful_lemmas = lemma_candidates

    # If we have only one meaningful lemma, return it
    if len(meaningful_lemmas) == 1:
        return meaningful_lemmas[0]

    # Score each lemma based on length and similarity to surface form
    scored_lemmas = []
    for lemma in meaningful_lemmas:
        # Length score (prefer longer lemmas)
        length_score = len(lemma) / max(len(l) for l in meaningful_lemmas)

        # Similarity score (prefer lemmas that share more characters with surface form)
        common_chars = len(set(lemma.lower()) & set(surface_form.lower()))
        similarity_score = common_chars / max(len(surface_form), len(lemma))

        # Combined score
        total_score = length_score * 0.7 + similarity_score * 0.3
        scored_lemmas.append((lemma, total_score))

    # Return the highest scoring lemma
    best_lemma = max(scored_lemmas, key=lambda x: x[1])[0]
    return best_lemma


def _is_meaningful_passive_form(surface_form: str, lemma_candidates: List[str]) -> bool:
    """Check if a passive form should be preserved because it has distinct meaning"""

    # Meaningful passive forms that should be preserved (specific list only)
    meaningful_passives = {
        "kazanılmış",  # earned (vs kazan = cauldron)
        "yapılmış",  # made/done
        "alınmış",  # taken
        "verilmiş",  # given
        "bulunmuş",  # found
        "görülmüş",  # seen
        "duyulmuş",  # heard
        "bilinmiş",  # known
        "sevilmiş",  # loved
        "sayılmış",  # counted/respected
        "doldurmamış",  # not filled/verb
        "edinilen",  # done/verb
        "edinilmiş",  # done/verb
        # Add more as needed - but be specific!
    }

    surface_lower = surface_form.lower()

    # Only check the explicit list - no heuristics that could interfere
    return surface_lower in meaningful_passives


def _normalize_unk_word(word: str) -> str:
    """Normalize UNK words - try to extract meaningful lemmas"""
    
    # Special UNK normalizations
    unk_normalizations = {
        'altsoyları': 'altsoy',  # altsoyları -> altsoy
        'üstsoyları': 'üstsoy',  # üstsoyları -> üstsoy
        # Add more as needed
    }
    
    word_lower = word.lower()
    if word_lower in unk_normalizations:
        return unk_normalizations[word_lower]
    
    # Try to detect plural forms and remove them
    if word_lower.endswith('ları') or word_lower.endswith('leri'):
        # Remove plural suffix and check if it makes sense
        base_word = word[:-4]  # Remove 'ları' or 'leri'
        if len(base_word) >= 4:  # Ensure reasonable length
            return base_word
    
    # If no normalization found, return original
    return word


def _get_corrected_lemma(surface_form: str, incorrect_lemma: str) -> str:
    """Get the corrected lemma for known incorrect cases"""
    
    # Special corrections where we need a specific lemma, not just the original word
    special_corrections = {
        'altsoyu': 'altsoy',  # altsoyu -> altsoy (not altso)
        # Add more special cases as needed
    }
    
    surface_lower = surface_form.lower()
    if surface_lower in special_corrections:
        return special_corrections[surface_lower]
    
    # For other incorrect lemmatizations, just return the original word
    return surface_form


def _should_correct_lemma(surface_form: str, lemma: str) -> bool:
    """Check if the lemmatization is incorrect and should be corrected"""

    # Known incorrect lemmatizations that should use the original form
    incorrect_lemmas = {
        "hurda": "hurd",  # hurda should remain hurda, not become hurd
        'ana': 'an',      # ana should remain ana, not become an
        'baba': 'bab',    # baba should remain baba, not become bab
        'mama': 'mam',    # mama should remain mama, not become mam
        'papa': 'pap',    # papa should remain papa, not become pap
        'dede': 'ded',    # dede should remain dede, not become ded
        'nene': 'nen',    # nene should remain nene, not become nen
        'altsoyu': 'altso',  # altsoyu should become altsoy, not altso
        # Add more known incorrect cases here as they are discovered
    }

    # Check if this is a known incorrect lemmatization
    surface_lower = surface_form.lower()
    lemma_lower = lemma.lower()

    if surface_lower in incorrect_lemmas:
        if incorrect_lemmas[surface_lower] == lemma_lower:
            return True  # This is a known incorrect lemmatization

    # Additional heuristics for detecting incorrect lemmatizations
    # If lemma is significantly shorter and doesn't make sense linguistically
    if len(lemma) < len(surface_form) - 2 and len(lemma) < 5:
        # Check if removing common suffixes would result in the lemma
        # This catches cases like "hurda" -> "hurd" where "a" is incorrectly treated as suffix
        common_suffixes = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]
        for suffix in common_suffixes:
            if (
                surface_form.lower().endswith(suffix)
                and surface_form.lower()[:-1] == lemma_lower
            ):
                # If the word is likely a standalone word (not a real inflection)
                if _is_likely_standalone_word(surface_form):
                    return True

    return False


def _is_likely_standalone_word(word: str) -> bool:
    """Check if a word is likely a standalone word rather than an inflected form"""

    # Common standalone words that might be misanalyzed
    standalone_words = {
        "hurda",
        "karda",
        "yurda",
        "barda",
        "perde",
        "erde",
        "çerde",
        # Add more as needed
    }

    return word.lower() in standalone_words


def _should_preserve_legal(surface_form: str, lemma: str, pos: str) -> bool:
    """Legal mode: preserve adjectives, proper nouns, and institutional terms"""

    # Always preserve adjectives
    if pos == "Adj":
        return True

    # Always preserve proper nouns
    if pos == "Propn":
        return True

    # Institutional terms are handled separately now

    # Check for potential adjectives based on morphological patterns
    if _is_likely_adjective(surface_form, lemma):
        return True

    # For compound nouns, preserve if lemma is significantly shorter (real compounds)
    if (
        pos == "Noun"
        and len(lemma) < len(surface_form) * 0.5
        and len(surface_form) > 10
    ):
        return True

    # Don't skip verbs completely - some might need preservation
    # But allow normal lemmatization for most verbs

    # For other cases, use lemma
    return False


def _is_institutional_term(surface_form: str, lemma: str) -> bool:
    """Check if a word is an institutional/official term that should be normalized to base form"""

    # Official institution suffixes - normalize these to base form (lığı)
    institutional_suffixes = [
        "lığa",
        "liğe",
        "luğa",
        "lüğe",  # bakanlığa -> bakanlığı
        "lığın",
        "liğin",
        "luğun",
        "lüğün",  # bakanlığın -> bakanlığı
        "lığından",
        "liğinden",
        "luğundan",
        "lüğünden",  # bakanlığından -> bakanlığı
        "lığına",
        "liğine",
        "luğuna",
        "lüğüne",  # bakanlığına -> bakanlığı
        "lığınca",
        "liğince",
        "luğunca",
        "lüğünce",  # bakanlığınca -> bakanlığı
    ]

    # Check if surface form ends with institutional suffixes that need normalization
    surface_lower = surface_form.lower()
    for suffix in institutional_suffixes:
        if surface_lower.endswith(suffix):
            # Additional check: lemma should end with 'lık/lik/luk/lük'
            if lemma.lower().endswith(("lık", "lik", "luk", "lük")):
                return True

    # Base forms (lığı, liği, luğu, lüğü) should be preserved as is
    base_suffixes = ["lığı", "liği", "luğu", "lüğü"]
    for suffix in base_suffixes:
        if surface_lower.endswith(suffix):
            if lemma.lower().endswith(("lık", "lik", "luk", "lük")):
                return True

    return False


def _normalize_institutional_term(surface_form: str, lemma: str) -> str:
    """Normalize institutional terms to their base form (lığı)"""

    # Map various institutional suffixes to base form
    suffix_mappings = {
        "lığa": "lığı",
        "liğe": "liği",
        "luğa": "luğu",
        "lüğe": "lüğü",
        "lığın": "lığı",
        "liğin": "liği",
        "luğun": "luğu",
        "lüğün": "lüğü",
        "lığından": "lığı",
        "liğinden": "liği",
        "luğundan": "luğu",
        "lüğünden": "lüğü",
        "lığına": "lığı",
        "liğine": "liği",
        "luğuna": "luğu",
        "lüğüne": "lüğü",
        "lığınca": "lığı",
        "liğince": "liği",
        "luğunca": "luğu",
        "lüğünce": "lüğü",
    }

    surface_lower = surface_form.lower()

    # Find the suffix and replace with base form
    for old_suffix, new_suffix in suffix_mappings.items():
        if surface_lower.endswith(old_suffix):
            # Replace the suffix
            base_form = surface_form[: -len(old_suffix)] + new_suffix
            return base_form

    # If no mapping found, return original
    return surface_form


def _is_likely_adjective(surface_form: str, lemma: str) -> bool:
    """Check if a word is likely an adjective based on morphological analysis only"""

    # Check for nispet eki (-i/-î): if surface ends with i/î and lemma doesn't
    if (
        surface_form.lower().endswith(("i", "î"))
        and not lemma.lower().endswith(("i", "î"))
        and len(surface_form) > len(lemma)
    ):

        # If the difference is exactly the -i/-î ending, likely an adjective
        if lemma + "i" == surface_form.lower() or lemma + "î" == surface_form.lower():
            return True

    # Check for other common adjective suffixes
    adjective_suffixes = [
        ("li", "l"),
        ("lı", "l"),
        ("lu", "l"),
        ("lü", "l"),  # possessive adjectives
        ("siz", "s"),
        ("sız", "s"),
        ("suz", "s"),
        ("süz", "s"),  # privative adjectives
    ]

    for suffix, stem_end in adjective_suffixes:
        if surface_form.lower().endswith(suffix) and lemma.endswith(stem_end):
            return True

    return False


def zemberek_analyze(text: str) -> List[Dict[str, Any]]:
    """Get detailed analysis using Zemberek"""
    if not _zemberek_morphology:
        raise ValueError("Zemberek not initialized")

    try:
        # Analyze and disambiguate the text
        analysis = _zemberek_morphology.analyzeAndDisambiguate(text)
        best_analysis = analysis.bestAnalysis()

        # Extract detailed analysis
        analyses = []
        for i in range(best_analysis.size()):
            analysis_item = best_analysis.get(i)
            lemma_list = analysis_item.getLemmas()

            analysis_dict = {
                "surface_form": str(analysis_item.surfaceForm()),
                "lemmas": (
                    [str(lemma_list.get(j)) for j in range(lemma_list.size())]
                    if lemma_list
                    else []
                ),
                "pos": (
                    str(analysis_item.getPos())
                    if hasattr(analysis_item, "getPos")
                    else "Unknown"
                ),
                "morphemes": (
                    [
                        str(analysis_item.getMorphemes().get(j))
                        for j in range(analysis_item.getMorphemes().size())
                    ]
                    if hasattr(analysis_item, "getMorphemes")
                    else []
                ),
            }
            analyses.append(analysis_dict)

        return analyses
    except Exception as e:
        log.error(f"Zemberek analysis failed: {e}")
        raise


class LemReq(BaseModel):
    texts: Union[str, List[str]]  # Single text or list of n-grams
    return_details: bool = False


@app.on_event("startup")
async def warmup():
    try:
        log.info("Initializing Zemberek...")
        init_zemberek()
        log.info("Zemberek warmup complete.")
    except Exception as e:
        log.exception("Warmup failed: %s", e)
        raise


@app.get("/health")
def health():
    return {"ok": True, "engine": "zemberek"}


@app.post("/lemmatize")
def lemmatize(req: LemReq) -> Dict[str, Any]:
    """Lemmatize single text or multiple n-grams (bigram, trigram, etc.)"""
    try:
        # Handle both single string and list of strings
        if isinstance(req.texts, str):
            texts_to_process = [req.texts]
        else:
            texts_to_process = req.texts

        if not texts_to_process:
            return {"lemmas": [], "details": []}

        lemmas = []
        details = []

        for text in texts_to_process:
            try:
                # Lemmatize each text (n-gram) separately
                lemmatized_text = zemberek_lemmatize_text(text)
                lemmas.append(lemmatized_text)

                if req.return_details:
                    analyses = zemberek_analyze(text)
                    details.append(
                        {
                            "input": text,
                            "lemma": lemmatized_text,
                            "analyses": analyses,
                            "engine": "zemberek-legal",
                        }
                    )
            except Exception as e:
                log.warning(f"Failed to lemmatize text '{text}': {e}")
                lemmas.append(text)  # Fallback to original
                if req.return_details:
                    details.append(
                        {
                            "input": text,
                            "lemma": text,
                            "analyses": [],
                            "engine": "fallback",
                            "error": str(e),
                        }
                    )

        result = {"lemmas": lemmas}
        if req.return_details:
            result["details"] = details

        return result
    except Exception as e:
        log.exception("lemmatize failed: %s", e)
        raise HTTPException(status_code=500, detail=f"lemmatize_failed: {e}")


@app.get("/")
def root():
    return {
        "service": "TR Lemmatizer",
        "engine": "Zemberek",
        "version": "2.0.0",
        "description": "Turkish lemmatization (preserves adjectives)",
        "endpoints": {"health": "/health", "lemmatize": "/lemmatize"},
        "usage": {
            "single_text": '{"texts": "merkezi sistem yönetimi"}',
            "multiple_ngrams": '{"texts": ["merkezi sistem", "idari yaptırım", "kanuni faiz"]}',
        },
    }
