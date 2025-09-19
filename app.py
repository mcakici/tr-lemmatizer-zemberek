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
                # Get the best lemma based on context
                best_lemma = str(lemma_list.get(0))
                surface_form = str(analysis_item.surfaceForm())
                pos = (
                    str(analysis_item.getPos())
                    if hasattr(analysis_item, "getPos")
                    else ""
                )

                # Legal mode: preserve adjectives and proper nouns
                if _should_preserve_legal(surface_form, best_lemma, pos):
                    lemmas.append(surface_form)  # Keep original form
                else:
                    lemmas.append(best_lemma)
            else:
                # Fallback to the original token
                lemmas.append(str(analysis_item.surfaceForm()))

        return " ".join(lemmas)
    except Exception as e:
        log.error(f"Zemberek lemmatization failed: {e}")
        raise


def _should_preserve_legal(surface_form: str, lemma: str, pos: str) -> bool:
    """Legal mode: preserve adjectives and proper nouns based on morphological analysis"""

    # Always preserve adjectives
    if pos == "Adj":
        return True

    # Always preserve proper nouns
    if pos == "Propn":
        return True

    # Check for potential adjectives based on morphological patterns
    if _is_likely_adjective(surface_form, lemma):
        return True

    # For compound nouns, preserve if lemma is significantly shorter
    if pos == "Noun" and len(lemma) < len(surface_form) * 0.7:
        return True

    # For other cases, use lemma
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
