"""Retrieval-augmented answering for the IRCTC help bot.

Two answer paths:
  * llm      — Claude, strictly grounded in the top-4 retrieved chunks.
  * offline  — deterministic extractive bullets from the same chunks, used when
               no valid API key is configured (or the API call fails).

Both paths return source links and never invent facts outside the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

import config
import vectorstore
from guard import Lang, detect_language

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0900-\u097F]+")
SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+|\n+")

STOPWORDS = {
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "this", "that", "these", "those", "to", "of", "in", "on", "at", "for",
    "with", "and", "or", "but", "if", "then", "than", "so", "can", "could", "would",
    "should", "will", "shall", "may", "might", "must", "have", "has", "had", "get",
    "got", "please", "tell", "know", "want", "need", "about", "what", "which", "who",
    "when", "where", "why", "how", "there", "here", "any", "some", "all", "also",
    "from", "by", "as", "not", "no", "yes", "am", "us", "one", "two", "much", "many",
    # Hindi
    "का", "की", "के", "को", "में", "से", "पर", "और", "या", "है", "हैं", "था", "थी",
    "थे", "हो", "होता", "होती", "कर", "करें", "करना", "करने", "किया", "कैसे", "क्या",
    "कब", "कहाँ", "कौन", "क्यों", "मैं", "मेरा", "मेरी", "मेरे", "आप", "आपका", "यह",
    "वह", "इस", "उस", "लिए", "नहीं", "भी", "तो", "कि", "जा", "जाता", "जाती", "सकता",
    "सकते", "सकती", "कोई", "कुछ", "एक", "दो", "बताओ", "बताइए", "जानकारी", "चाहिए",
}

UNSURE_BY_LANG = {"en": config.UNSURE, "hi": config.UNSURE_HI}


def tokens(text: str) -> set[str]:
    out = set()
    for tok in TOKEN_RE.findall((text or "").lower()):
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        out.add(tok)
    return out


def token_match(a: str, b: str) -> bool:
    """Exact match, or a shared prefix long enough to catch cancel/cancellation."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long.startswith(short) and len(long) - len(short) <= 6


def overlap_count(query: set[str], text: set[str]) -> int:
    if not query or not text:
        return 0
    return sum(1 for q in query if any(token_match(q, t) for t in text))


@dataclass
class Retrieved:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    lex: int = 0

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def source_title(self) -> str:
        return str(self.metadata.get("source_title") or self.metadata.get("title") or "IRCTC")

    @property
    def heading(self) -> str:
        return str(self.metadata.get("heading", ""))


class RAGEngine:
    def __init__(
        self,
        embed_model: str | None = None,
        collection_name: str | None = None,
        offline: bool | None = None,
        top_k: int | None = None,
    ) -> None:
        self.embed_model_name = embed_model or config.EMBED_MODEL
        self.index_name = collection_name or config.INDEX_NAME
        self.offline = config.OFFLINE if offline is None else offline
        self.top_k = config.TOP_K if top_k is None else top_k
        self._model: SentenceTransformer | None = None
        self._llm_failed = False
        self._index = None
        self._chunks: list[vectorstore.StoredChunk] | None = None

    # ------------------------------------------------------------------ #
    # Lazy resources
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embed_model_name)
        return self._model

    @property
    def store(self) -> tuple[object, list[vectorstore.StoredChunk]]:
        if self._index is None or self._chunks is None:
            if not vectorstore.index_exists():
                raise FileNotFoundError(
                    "Vector store not found. Run `python ingest.py` first."
                )
            self._index, self._chunks = vectorstore.load_index()
        return self._index, self._chunks

    @property
    def index(self):
        return self.store[0]

    @property
    def chunks(self) -> list[vectorstore.StoredChunk]:
        return self.store[1]

    def count(self) -> int:
        try:
            return int(self.index.ntotal)
        except Exception:
            return 0

    def health(self) -> dict[str, Any]:
        return {
            "chunks": self.count(),
            "embedding_model": self.embed_model_name,
            "collection": self.index_name,
            "backend": "faiss",
            "llm": (
                "offline"
                if self.offline or not config.LLM_ENABLED
                else ("claude-unavailable-fallback" if self._llm_failed else "claude")
            ),
        }

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _lexical_matches(self, query: str, limit: int = 12) -> list[tuple[int, "vectorstore.StoredChunk"]]:
        """Keyword matches over the whole (small) corpus.

        Embeddings handle paraphrase; this catches rare tokens such as "3A",
        "TDR" or "आईडी" that can otherwise be diluted in a long chunk.
        """
        q_tokens = tokens(query)
        if not q_tokens:
            return []
        scored: list[tuple[int, "vectorstore.StoredChunk"]] = []
        for chunk in self.chunks:
            lex = overlap_count(
                q_tokens, tokens(str(chunk.metadata.get("heading", "")) + " " + chunk.text)
            )
            if lex >= 2:
                scored.append((lex, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    def _query(self, query: str, k: int, lang: Lang | None) -> list[Retrieved]:
        embedding = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        index, chunks = self.store
        if index.ntotal == 0:
            return []

        # Fetch extra candidates so a language filter still has choice.
        fetch_k = min(int(index.ntotal), max(k * 8, 24))
        scores, ids = index.search(np.asarray(embedding, dtype="float32"), fetch_k)

        pool: dict[str, Retrieved] = {}
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[int(idx)]
            pool[chunk.id] = Retrieved(
                id=chunk.id,
                text=chunk.text,
                score=float(score),
                metadata=chunk.metadata,
            )

        for lex, chunk in self._lexical_matches(query):
            if chunk.id in pool:
                pool[chunk.id].lex = lex
            else:
                pool[chunk.id] = Retrieved(
                    id=chunk.id,
                    text=chunk.text,
                    score=0.30,
                    metadata=chunk.metadata,
                    lex=lex,
                )

        items = sorted(
            pool.values(), key=lambda x: x.score + 0.03 * x.lex, reverse=True
        )

        if lang:
            preferred = [i for i in items if i.metadata.get("lang") == lang]
            if len(preferred) >= k:
                return preferred[:k]
            # Not enough material in that language: mix the other language in.
            seen = {i.id for i in preferred}
            return (preferred + [i for i in items if i.id not in seen])[:k]

        return items[:k]

    def retrieve(self, query: str, lang: Lang | None = None, k: int | None = None) -> list[Retrieved]:
        return self._query(query, k or self.top_k, lang)

    # ------------------------------------------------------------------ #
    # Public answer API
    # ------------------------------------------------------------------ #
    def answer(
        self,
        message: str,
        lang: str = "auto",
        history: list[dict[str, str]] | None = None,
        force_offline: bool | None = None,
    ) -> dict[str, Any]:
        message = (message or "").strip()
        language: Lang = detect_language(message) if lang in ("auto", "", None) else lang  # type: ignore[assignment]
        if language not in ("en", "hi"):
            language = detect_language(message)

        use_offline = (
            force_offline
            if force_offline is not None
            else (self.offline or not config.LLM_ENABLED)
        )
        # Claude gets the required top-4 context. The offline answerer looks at a
        # few more chunks so it can pick the best FAQ section instead of a stray
        # bullet from an adjacent topic.
        retrieve_k = max(self.top_k, 8) if use_offline else self.top_k
        chunks = self.retrieve(message, lang=language, k=retrieve_k)
        best = chunks[0].score if chunks else 0.0
        query_tokens = tokens(message)
        best_overlap = max(
            (overlap_count(query_tokens, tokens(c.heading + " " + c.text)) for c in chunks),
            default=0,
        )
        relevant = bool(chunks) and (best >= config.MIN_SCORE or best_overlap >= 2)

        if not relevant:
            return {
                "answer": UNSURE_BY_LANG[language],
                "language": language,
                "sources": [],
                "mode": "offline" if (force_offline or self.offline or not config.LLM_ENABLED) else "claude",
                "retrieved": [c.id for c in chunks],
                "confident": False,
            }

        used = chunks
        if use_offline:
            text, used = self._extractive_answer(message, chunks, language)
            mode = "offline"
        else:
            text, mode, used = self._llm_answer(message, chunks, language, history or [])

        return {
            "answer": text,
            "language": language,
            "sources": self._sources(used or chunks),
            "mode": mode,
            "retrieved": [c.id for c in chunks],
            "confident": True,
        }

    # ------------------------------------------------------------------ #
    # Offline extractive answer
    # ------------------------------------------------------------------ #
    # Query intent → corpus category, used to prefer the right document set.
    INTENT_CATEGORIES: dict[str, str] = {
        "refund": "refund",
        "tatkal": "tatkal",
        "id": "id-rules",
        "class": "classes-quotas",
        "quota": "classes-quotas",
        "pnr": "pnr",
        "booking": "booking",
    }

    @staticmethod
    def _intents(query: str) -> set[str]:
        q = (query or "").lower()
        found: set[str] = set()
        if re.search(r"refund|cancel|कैंसिल|रद्द|रिफंड|tdr|क्लर्केज", q):
            found.add("refund")
        if re.search(r"tatkal|तत्काल", q):
            found.add("tatkal")
        if re.search(r"\bid\b|identity|proof|document|आईडी|पहचान|दस्तावे", q):
            found.add("id")
        if re.search(
            r"\bclass|\bsl\b|3a\b|2a\b|1a\b|3e\b|\bcc\b|\bec\b|sleeper|"
            r"chair\s*car|first\s*ac|\d\s*tier|क्लास|श्रेणी|3ए|स्लीपर",
            q,
        ):
            found.add("class")
        if re.search(r"quota|कोटा", q):
            found.add("quota")
        if re.search(r"\bpnr\b", q):
            found.add("pnr")
        if re.search(r"\bbook|booking|reserve|बुक|आरक्षण", q):
            found.add("booking")
        return found

    @staticmethod
    def _section_bullets(chunk: "Retrieved") -> list[str]:
        lines = chunk.text.split("\n")
        if lines and " — " in lines[0] and len(lines[0]) < 160:
            lines = lines[1:]
        bullets: list[str] = []
        prose: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("- "):
                bullets.append(line[2:].strip())
            elif len(line) > 40:
                prose.append(line)
        if prose and bullets:
            return [prose[0]] + bullets
        return bullets or prose

    def _rank_sections(
        self, query: str, chunks: list[Retrieved], language: Lang = "en"
    ) -> list[tuple[float, Retrieved]]:
        q_tokens = tokens(query)
        intents = self._intents(query)
        ranked: list[tuple[float, Retrieved]] = []
        for chunk in chunks:
            heading_tokens = tokens(chunk.heading)
            body_tokens = tokens(chunk.text)
            heading_lex = overlap_count(q_tokens, heading_tokens)
            body_lex = overlap_count(q_tokens, body_tokens)
            bullets = self._section_bullets(chunk)
            max_bullet_lex = max(
                (overlap_count(q_tokens, tokens(b)) for b in bullets), default=0
            )
            score = (
                chunk.score
                + heading_lex * 0.18
                + min(body_lex, 6) * 0.015
                + chunk.lex * 0.02
                + max_bullet_lex * 0.09
            )
            asks_charge = bool(
                re.search(r"charge|fee|fare|cost|चार्ज|शुल्क|किराया|कीमत", query, re.I)
            )
            if not asks_charge and re.search(
                r"charges?|fees?|चार्ज|शुल्क", chunk.heading, re.I
            ):
                score -= 0.15
            if re.search(r"\bhow\b|steps?|process|procedure|कैसे|प्रक्रिया|चरण", query, re.I) and re.search(
                r"\bhow\b|steps?|process|procedure|guide|where|check|कैसे|प्रक्रिया|चरण|देखें|कहाँ", chunk.heading, re.I
            ):
                score += 0.25
            if re.search(r"forget|forgot|missing|lost|without|not carry|भूल|खो", query, re.I) and re.search(
                r"missing|if |without|न हो|नहीं", chunk.heading, re.I
            ):
                score += 0.6
            heading_lower = chunk.heading.lower()
            intent_words = {
                "id": ("id", "identity", "आईडी", "पहचान"),
                "class": ("class", "क्लास", "श्रेणी"),
                "quota": ("quota", "कोटा"),
                "pnr": ("pnr",),
                "refund": ("cancel", "refund", "tdr", "रद्द", "रिफंड", "कैंसिल"),
            }
            if any(
                any(word in heading_lower for word in intent_words.get(intent, ()))
                for intent in intents
            ):
                score += 0.25
            category = str(chunk.metadata.get("category", ""))
            if category in {self.INTENT_CATEGORIES[i] for i in intents}:
                score += 0.12
            chunk_lang = str(chunk.metadata.get("lang", "en"))
            if chunk_lang == language:
                score += 0.15 if language == "hi" else 0.04
            ranked.append((score, chunk))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    @staticmethod
    def _rank_bullets(
        bullets: list[str], query: str, intents: set[str]
    ) -> list[str]:
        """Order bullets by how well they cover the question's aspects."""
        q_tokens = tokens(query)
        asks_charge = bool(re.search(r"charge|fee|fare|cost|चार्ज|शुल्क|किराया|कीमत", query, re.I))
        asks_time = bool(re.search(r"time|timing|when|बजे|समय|कब", query, re.I))
        asks_class = "class" in intents

        # Turn colloquial class descriptions into the code used in the docs.
        q_lower = query.lower()
        class_code = ""
        for pattern, code in [
            (r"\b1a\b|first\s*ac|ac\s*first", "1A"),
            (r"\b2a\b|2\s*tier|second\s*ac|ac\s*2", "2A"),
            (r"\b3a\b|3\s*tier|third\s*ac|ac\s*3", "3A"),
            (r"\b3e\b|3\s*economy", "3E"),
            (r"\bsl\b|sleeper", "SL"),
            (r"\bcc\b|chair\s*car", "CC"),
            (r"\bec\b|executive", "EC"),
        ]:
            if re.search(pattern, q_lower):
                class_code = code
                break

        scored: list[tuple[float, int, str]] = []
        for position, bullet in enumerate(bullets):
            b_tokens = tokens(bullet)
            score = overlap_count(q_tokens, b_tokens) * 1.2
            if asks_charge and re.search(r"₹|\bGST\b|%", bullet, re.I):
                score += 1.2
            if asks_time and re.search(r"\d{1,2}[:.]\d{2}|बजे", bullet):
                score += 1.2
            if asks_class and re.search(r"\b(1A|2A|3A|3E|SL|CC|EC|2S|FC)\b", bullet):
                score += 1.0
            if class_code:
                class_pattern = rf"\b{class_code}\b"
                if class_code == "3A":
                    class_pattern += r"|3\s*tier"
                elif class_code == "2A":
                    class_pattern += r"|2\s*tier"
                elif class_code == "1A":
                    class_pattern += r"|first\s*ac"
                elif class_code == "SL":
                    class_pattern += r"|sleeper"
                if re.search(class_pattern, bullet, re.I):
                    score += 4.0
            scored.append((score, position, bullet))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [b for _, _, b in scored]

    def _extractive_answer(
        self, query: str, chunks: list[Retrieved], language: Lang
    ) -> str:
        ranked = self._rank_sections(query, chunks, language)
        if not ranked:
            return UNSURE_BY_LANG[language], []

        best_score, best_chunk = ranked[0]
        selected = [best_chunk]
        # Add a second, nearly-as-good section for multi-part questions
        # (e.g. "Tatkal timing and charges"), but never more than two.
        if len(ranked) > 1:
            second_score, second_chunk = ranked[1]
            if second_score >= best_score * 0.88 and second_chunk.id != best_chunk.id:
                selected.append(second_chunk)

        intents = self._intents(query)
        is_list_question = bool(
            re.search(r"which|what.*(id|document)|list|difference|types?|कौन|कौन\s*सा|सूची|अंतर|प्रकार", query, re.I)
        )
        limit = 10 if ("id" in intents or is_list_question) else 6
        groups = {(c.source, c.heading) for c in selected}
        per_section = 3 if len(groups) > 1 else limit

        preserve_order = bool(
            re.search(r"how|steps?|process|procedure|कैसे|प्रक्रिया|चरण", query, re.I)
        ) or "id" in intents

        bullets: list[str] = []
        seen: set[str] = set()
        for chunk in selected:
            section = self._section_bullets(chunk)
            if not preserve_order:
                section = self._rank_bullets(section, query, intents)
            taken = 0
            for bullet in section:
                key = re.sub(r"\W+", " ", bullet.lower())[:90]
                if key in seen:
                    continue
                seen.add(key)
                bullets.append(bullet)
                taken += 1
                if taken >= per_section or len(bullets) >= limit:
                    break
            if len(bullets) >= limit:
                break

        if not bullets:
            opening = " ".join(chunks[0].text.split("\n")[1:])[:500]
            bullets = [opening] if opening else []

        if not bullets:
            return UNSURE_BY_LANG[language], []

        lead = ""
        if language == "hi" and not any(c.metadata.get("lang") == "hi" for c in selected):
            lead = "इस विषय की हिंदी जानकारी उपलब्ध नहीं मिली; अंग्रेज़ी में यही जानकारी:\n"
        elif language == "en" and all(c.metadata.get("lang") == "hi" for c in selected):
            lead = "The matching source is in Hindi; here is the information:\n"

        heading = best_chunk.heading.strip()
        if heading and len(heading) < 80 and not lead:
            lead = f"{heading}:\n"

        rendered = []
        for bullet in bullets:
            text = re.sub(r"\s+", " ", bullet).strip()
            if len(text) > 280:
                text = text[:277].rsplit(" ", 1)[0] + "…"
            rendered.append(f"- {text}")
        return lead + "\n".join(rendered), selected

    # ------------------------------------------------------------------ #
    # Claude answer
    # ------------------------------------------------------------------ #
    def _llm_answer(
        self,
        message: str,
        chunks: list[Retrieved],
        language: Lang,
        history: list[dict[str, str]],
    ) -> tuple[str, str, list[Retrieved]]:
        if self._llm_failed:
            text, used = self._extractive_answer(message, chunks, language)
            return text, "offline-fallback", used
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            context_blocks = []
            for i, chunk in enumerate(chunks, start=1):
                context_blocks.append(
                    f"[{i}] ({chunk.source_title} — {chunk.source})\n{chunk.text}"
                )
            context = "\n\n".join(context_blocks)

            language_name = "Hindi (Devanagari)" if language == "hi" else "English"
            system = (
                "You are an unofficial IRCTC help assistant. You explain public IRCTC/Indian "
                "Railways rules in short, simple point form. You are not IRCTC, you cannot "
                "access accounts, and you never perform transactions.\n\n"
                "STRICT RULES:\n"
                "- Use ONLY the CONTEXT below. Do not use outside knowledge or guess.\n"
                f"- If the context does not answer the question, reply exactly: "
                f"{UNSURE_BY_LANG[language]}\n"
                f"- Reply in {language_name}, matching the user's language.\n"
                "- Use 3 to 6 bullets, each one short idea, plain words, max 25 words.\n"
                "- Include specific numbers, timings, class codes and rupee amounts when the "
                "context gives them.\n"
                "- Never ask for or repeat passwords, OTPs, card numbers, CVV, UPI IDs or Aadhaar.\n"
                "- Never claim to book, cancel, change a ticket, or show live PNR/seat/fare/train data.\n"
                "- Do not add a disclaimer, a preamble or a closing question; the UI shows them.\n"
                "- Do not write URLs; the app adds source links.\n"
            )

            messages: list[dict[str, str]] = []
            for turn in history[-4:]:
                role = turn.get("role")
                content = (turn.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content[:1500]})
            messages.append(
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION: {message}",
                }
            )

            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=500,
                temperature=0.2,
                system=system,
                messages=messages,
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            ).strip()
            if not text:
                raise ValueError("empty response")
            return text, "claude", chunks
        except Exception as exc:  # noqa: BLE001 - any API failure falls back safely
            self._llm_failed = True
            print(f"[rag] Claude unavailable ({exc.__class__.__name__}); using offline answerer")
            text, used = self._extractive_answer(message, chunks, language)
            return text, "offline-fallback", used

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sources(chunks: list[Retrieved], limit: int = 3) -> list[dict[str, str]]:
        seen: set[str] = set()
        out: list[dict[str, str]] = []
        for chunk in chunks:
            url = chunk.source
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"title": chunk.source_title, "url": url})
            if len(out) >= limit:
                break
        return out
