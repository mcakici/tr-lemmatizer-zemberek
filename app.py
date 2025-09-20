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
                    lemma_candidates = [str(lemma_list.get(j)) for j in range(lemma_list.size())]
                    best_lemma = _select_best_lemma(lemma_candidates, str(analysis_item.surfaceForm()))
                surface_form = str(analysis_item.surfaceForm())
                pos = (
                    str(analysis_item.getPos())
                    if hasattr(analysis_item, "getPos")
                    else ""
                )

                # Handle UNK tokens - keep original word as is, but continue processing other words
                if best_lemma == "UNK" or best_lemma == "Unknown":
                    lemmas.append(surface_form)  # Keep UNK word as original
                # Check for incorrect lemmatization corrections
                elif _should_correct_lemma(surface_form, best_lemma):
                    lemmas.append(surface_form)  # Use original form for incorrect lemmas
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


def _should_correct_lemma(surface_form: str, lemma: str) -> bool:
    """Check if the lemmatization is incorrect and should be corrected"""
    
    # Known incorrect lemmatizations that should use the original form
    incorrect_lemmas = {
        'hurda': 'hurd',  # hurda should remain hurda, not become hurd
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
        common_suffixes = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']
        for suffix in common_suffixes:
            if surface_form.lower().endswith(suffix) and surface_form.lower()[:-1] == lemma_lower:
                # If the word is likely a standalone word (not a real inflection)
                if _is_likely_standalone_word(surface_form):
                    return True
    
    return False


def _is_likely_standalone_word(word: str) -> bool:
    """Check if a word is likely a standalone word rather than an inflected form"""
    
    # Common standalone words that might be misanalyzed
    standalone_words = {
        'hurda', 'karda', 'yurda', 'barda', 'perde', 'erde', 'çerde',
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

    # Preserve institutional and official terms
    if _is_institutional_term(surface_form, lemma):
        return True

    # Check for potential adjectives based on morphological patterns
    if _is_likely_adjective(surface_form, lemma):
        return True

    # For compound nouns, preserve if lemma is significantly shorter (real compounds)
    if pos == "Noun" and len(lemma) < len(surface_form) * 0.5 and len(surface_form) > 10:
        return True

    # For other cases, use lemma
    return False


def _is_institutional_term(surface_form: str, lemma: str) -> bool:
    """Check if a word is an institutional/official term that should be preserved"""
    
    # Official institution suffixes that should be preserved
    institutional_suffixes = [
        'lığı', 'liği', 'luğu', 'lüğü',  # bakanlığı, müdürlüğü
        'lığa', 'liğe', 'luğa', 'lüğe',  # bakanlığa, müdürlüğe  
        'lığın', 'liğin', 'luğun', 'lüğün',  # bakanlığın, müdürlüğün
        'lığından', 'liğinden', 'luğundan', 'lüğünden',  # bakanlığından
        'lığına', 'liğine', 'luğuna', 'lüğüne',  # bakanlığına
    ]
    
    # Check if surface form ends with institutional suffixes
    surface_lower = surface_form.lower()
    for suffix in institutional_suffixes:
        if surface_lower.endswith(suffix):
            # Additional check: lemma should end with 'lık/lik/luk/lük'
            if lemma.lower().endswith(('lık', 'lik', 'luk', 'lük')):
                return True
    
    return False


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
        "description": "Turkish lemmatization with legal mode (preserves adjectives)",
        "endpoints": {"health": "/health", "lemmatize": "/lemmatize"},
        "usage": {
            "single_text": '{"texts": "merkezi sistem yönetimi"}',
            "multiple_ngrams": '{"texts": ["merkezi sistem", "idari yaptırım", "kanuni faiz"]}',
        },
    }
