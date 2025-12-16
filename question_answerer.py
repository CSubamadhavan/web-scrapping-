# question_answerer.py (complete)
import re
import traceback
from typing import List, Dict, Tuple, Optional

import torch
from pymilvus import connections, Collection
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer

from classifier.keybert_topic_extractor import KeyBERTTopicExtractor


# ---------- Basic extractors ----------
def extract_emails(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text or "")


def extract_phones(text: str) -> List[str]:
    return re.findall(r"\+?\d[\d\s\-]{8,}", text or "")


def extract_address(text: str) -> Optional[str]:
    patterns = [
        r"(?:Door\s*No\.?\s*\d+[,\s]*)?[A-Za-z0-9\s.,'-]+(?:Street|St|Road|Rd|Nagar|Colony|Layout|Avenue|Ave|Block|Phase|Floor|Building|Pillai|Colony)[,.\s]*[A-Za-z\s]+[,.\s-]*\d{5,6}",
        r"[A-Za-z0-9\s.,'-]+,\s*[A-Za-z\s]+[,.\s-]*\d{5,6}"
    ]
    for pat in patterns:
        m = re.search(pat, text or "", re.IGNORECASE)
        if m:
            addr = re.split(r"\s(?:phone|tel|email|mail)\b", m.group(0), flags=re.IGNORECASE)[0]
            addr = addr.strip().rstrip(".")
            # Fix: remove leading "address"
            addr = re.sub(r"^\s*address\s*[:,-]?\s*", "", addr, flags=re.IGNORECASE)
            return addr
    return None


# ---------- Text helpers ----------
def _sentences(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", t)
    bullets = re.findall(r"(?m)^\s*[-*•]\s+.*$", text or "")
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    for b in bullets:
        bb = re.sub(r"\s+", " ", b).strip()
        if bb and bb not in seen:
            out.append(bb)
    return out


def _tokenize(text: str, stop: Optional[set] = None) -> List[str]:
    stop = stop or set()
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", (text or "").lower())
    return [w for w in words if len(w) >= 3 and w not in stop]


def _expand_to_sentence_bounds(text: str, start: int, end: int) -> str:
    if not text:
        return ""
    start = max(0, start)
    end = min(len(text), end)
    left = max(text.rfind('.', 0, start), text.rfind('?', 0, start), text.rfind('!', 0, start))
    cut_l = 0 if left == -1 else left + 1
    right_candidates = [x for x in [text.find('.', end), text.find('?', end), text.find('!', end)] if x != -1]
    cut_r = min(right_candidates) + 1 if right_candidates else len(text)
    snip = text[cut_l:cut_r].strip()
    return re.sub(r"\s+", " ", snip)


# ---------- Intent detection ----------
LIST_INTENTS = {
    "services": ["services", "service list", "what services", "offerings", "capabilities", "solutions"],
    "team": ["team members", "crew", "members", "staff", "leadership", "founders", "people"],
    "features": ["features", "key features", "highlights", "benefits"],
    "products": ["products", "product list", "portfolio", "case studies", "clients"],
    "faq": ["faq", "faqs", "frequently asked questions"],
}


def _detect_list_intent(question: str) -> Optional[str]:
    q = (question or "").lower()
    for intent, syns in LIST_INTENTS.items():
        if any(s in q for s in syns):
            return intent
    if re.search(r"\bnames?\s+of\s+the\s+services?\b", q):
        return "services"
    return None


# ---------- Collect snippets ----------
def _collect_snippets(text: str, keywords: List[str], window: int = 360) -> List[str]:
    text = text or ""
    low = text.lower()
    hits = []
    for kw in keywords or []:
        k = kw.lower()
        start = 0
        while True:
            idx = low.find(k, start)
            if idx == -1:
                break
            left = max(0, idx - window // 2)
            right = min(len(text), idx + len(k) + window // 2)
            snip = _expand_to_sentence_bounds(text, left, right)
            if len(snip) >= 40:
                hits.append((1, snip))
            start = idx + len(k)
    hits.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    out, seen = [], set()
    for _, s in hits:
        key = s[:160].lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= 5:
            break
    return out


def _as_list_lines(text: str) -> List[str]:
    lines = []
    for m in re.finditer(r"(?m)^\s*[-*•]\s*(.+)$", text or ""):
        line = re.sub(r"\s+", " ", m.group(1)).strip(" -—–•:;")
        if 2 <= len(line) <= 140:
            lines.append(line)

    if not lines:
        for sent in re.split(r"[.\n]", text or ""):
            sent = sent.strip()
            if not sent:
                continue
            if sent.count(",") >= 2 and len(sent) <= 300:
                parts = [p.strip() for p in sent.split(",") if p.strip()]
                if 2 <= len(parts) <= 20:
                    lines.extend(parts)

    clean, seen = [], set()
    for l in lines:
        l2 = re.sub(r"\s{2,}", " ", l).strip(" -—–•:;")
        key = l2.lower()
        if key not in seen and 2 <= len(l2) <= 140:
            seen.add(key)
            clean.append(l2)

    return clean[:15]


# ---------- Main class ----------
class QuestionAnswerer:
    def __init__(self, collection_name="webpages", mongo_uri=None,
                 milvus_host="localhost", milvus_port=19530, collection=None):

        self.device = 0 if torch.cuda.is_available() else -1
        print("[INFO] Using", "GPU" if self.device == 0 else "CPU")

        # Milvus
        try:
            connections.connect(alias="default", host=milvus_host, port=milvus_port)
            self.collection = collection or Collection(collection_name)
            print(f"[✅] Connected to Milvus collection '{collection_name}'")
        except Exception as e:
            print(f"[WARNING] Milvus connection failed: {e}")
            self.collection = None

        # Mongo
        try:
            uri = mongo_uri or "mongodb://localhost:27017/"
            self.mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client["chatbotDB"]
            self.scraped_data_collection = self.db["scraped_data"]
            self.qa_collection = self.db["qa_pairs"]
            self.mongo_client.server_info()
            print("[✅] Connected to MongoDB")
        except Exception as e:
            print(f"[WARNING] MongoDB connection failed: {e}")
            self.scraped_data_collection = None
            self.qa_collection = None

        # Embedding model
        try:
            self.embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1",
                                                device=("cuda" if self.device == 0 else "cpu"))
            print("[✅] Embedding model loaded")
        except Exception as e:
            print("[❌] Embedding model load failed:", e)
            self.embedder = None

        # Extractive QA
        try:
            self.extractive_qa = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=self.device
            )
            print("[✅] Extractive QA model loaded")
        except Exception as e:
            print("[❌] Extractive QA load failed:", e)
            self.extractive_qa = None

        # Generative QA
        try:
            self.generative_qa = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=self.device
            )
            self.generative_model_name = "google/flan-t5-base"
            self.generative_tokenizer = AutoTokenizer.from_pretrained(self.generative_model_name)
            print("[✅] Generative QA model loaded")
        except Exception as e:
            print("[❌] Generative QA load failed:", e)
            self.generative_qa = None
            self.generative_tokenizer = None

        # Topic extractor (KeyBERT)
        try:
            self.topic_extractor = KeyBERTTopicExtractor()
            print("[✅] KeyBERT topic extractor ready")
        except Exception:
            self.topic_extractor = None
            print("[⚠️] KeyBERT topic extractor not available")

        # Config
        self.MIN_SIMILARITY_THRESHOLD = 0.3
        self.MIN_ANSWER_LENGTH = 10
        self.STOP = set("a an and the of for to in on with at by from as is are was were be been have has had "
                        "do does did your you me our their my we us it this that these those what which".split())
        self.CATEGORY_SYNONYMS = {
            "Policy": ["policy", "privacy", "terms"],
            "Contact": ["contact", "address", "phone", "email", "location", "office"],
            "Services": ["service", "services", "solution", "solutions", "offerings"],
            "Projects": ["project", "portfolio", "clients", "work"],
            "UI/UX": ["ui", "ux", "design", "user interface", "user experience"],
        }

    # -------- Contact helpers --------
    def extract_contact_details(self, text: str) -> Dict[str, object]:
        emails = extract_emails(text)
        phones = extract_phones(text)
        address = extract_address(text)
        return {
            "email": emails,
            "phone": phones,
            "address": address
        }

    def is_contact_query(self, question: str) -> bool:
        q = (question or "").lower()
        return any(k in q for k in ["address", "location", "contact", "phone", "email", "mobile", "call", "office", "mail"])

    def get_contact_query_type(self, question: str) -> str:
        """Determine the specific type of contact information requested"""
        q = (question or "").lower()

        # Check for email/mail queries
        email_keywords = ["email", "mail", "mail id", "mail-id", "@", "e-mail"]
        if any(k in q for k in email_keywords):
            return "email"

        # Check for phone queries
        phone_keywords = ["phone", "mobile", "contact number", "call", "telephone", "tel", "whatsapp"]
        if any(k in q for k in phone_keywords):
            return "phone"

        # Check for address queries
        address_keywords = ["address", "location", "office", "where", "situated", "located", "headquarters"]
        if any(k in q for k in address_keywords):
            return "address"

        # Default to all if can't determine specific type
        return "all"

    def format_contact_response(self, contact_info: Dict, query_type: str) -> str:
        """Format contact response based on query type"""
        parts = []

        if query_type == "email" or query_type == "all":
            if contact_info.get("email"):
                if query_type == "email":
                    return ", ".join(contact_info["email"])
                else:
                    parts.append("✉️ Email: " + ", ".join(contact_info["email"]))

        if query_type == "phone" or query_type == "all":
            if contact_info.get("phone"):
                if query_type == "phone":
                    return ", ".join(contact_info["phone"])
                else:
                    parts.append("📞 Phone: " + ", ".join(contact_info["phone"]))

        if query_type == "address" or query_type == "all":
            if contact_info.get("address"):
                if query_type == "address":
                    return contact_info["address"]
                else:
                    parts.append("📍 Address: " + contact_info["address"])

        return "\n".join(parts) if parts else "No contact details found."

    # ---------- Chunking ----------
    def _chunk_text(self, text: str, max_words: int = 260, stride: int = 110) -> List[str]:
        words = (text or "").split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i: i + max_words])
            chunks.append(chunk)
            if i + max_words >= len(words):
                break
            i += max(1, stride)
        return chunks[:12]

    def _extractive_qa_best(self, question: str, context: str) -> Tuple[str, float]:
        best, best_score = "", 0.0
        if not self.extractive_qa:
            return best, best_score
        for ch in self._chunk_text(context):
            try:
                res = self.extractive_qa(question=question, context=ch)
                ans, score = res.get("answer", ""), float(res.get("score", 0))
                if score > best_score and ans and ans.strip():
                    best, best_score = ans, score
            except Exception:
                continue
        return best, best_score

    # ---------- Scoring ----------
    def _question_categories(self, question: str) -> List[str]:
        q = (question or "").lower()
        hit = []
        for cat, syns in self.CATEGORY_SYNONYMS.items():
            if any(s in q for s in syns):
                hit.append(cat)
        return hit

    def _topic_overlap_boost(self, question_tokens: List[str], doc_topic: Optional[object]) -> float:
        """
        Small bonus if question tokens overlap with doc 'topic' field or KeyBERT topics.
        doc_topic can be a list of strings or a single string.
        """
        if not question_tokens:
            return 0.0
        toks = set(question_tokens)
        boost = 0.0
        if isinstance(doc_topic, list):
            topic_tokens = set()
            for t in doc_topic:
                topic_tokens.update(_tokenize(t))
            overlap = len(toks & topic_tokens)
            if overlap >= 1:
                boost += min(0.15, 0.05 * overlap)
        elif isinstance(doc_topic, str) and doc_topic.strip():
            topic_tokens = set(_tokenize(doc_topic))
            overlap = len(toks & topic_tokens)
            if overlap >= 1:
                boost += min(0.15, 0.05 * overlap)
        return boost

    def _score_doc(self, question_vec, tokens: List[str], categories_hit: List[str], doc: Dict) -> float:
        title = (doc.get("title") or "").lower()
        content = doc.get("content") or doc.get("lemmatized_content") or ""
        pred_cat = (doc.get("predicted_category") or "").lower()

        low = (title + " " + content).lower()
        coverage = 0.0
        if tokens:
            present = sum(1 for t in set(tokens) if t in low)
            coverage = present / len(set(tokens))

        cat_bonus = 0.0
        if categories_hit and pred_cat:
            if any(c.lower() == pred_cat for c in categories_hit):
                cat_bonus = 0.25
        if categories_hit:
            for cat in categories_hit:
                for syn in self.CATEGORY_SYNONYMS.get(cat, []):
                    if syn in title:
                        cat_bonus = max(cat_bonus, 0.20)

        # Topic bonus (from 'topic' field if available)
        topic_bonus = self._topic_overlap_boost(tokens, doc.get("topic"))

        ctx = content[:1500]
        try:
            if self.embedder:
                doc_vec = self.embedder.encode([ctx])[0]
                sem = cosine_similarity([question_vec], [doc_vec])[0][0]
            else:
                sem = 0.0
        except Exception:
            sem = 0.0

        t_hit = 0.1 if any(t in title for t in tokens[:5]) else 0.0
        return sem * 0.55 + coverage * 0.20 + cat_bonus + t_hit + topic_bonus

    # ---------- Section/list aware extraction ----------
    def _extract_list_answer(self, question: str, content: str) -> Tuple[str, str]:
        intent = _detect_list_intent(question)
        if not intent:
            return "", ""

        anchors_map = {
            "services": ["service", "services", "offerings", "capabilities", "solutions"],
            "team": ["team", "crew", "members", "leadership", "people", "founders"],
            "features": ["features", "benefits", "highlights", "key features"],
            "products": ["products", "portfolio", "case studies", "clients"],
            "faq": ["faq", "frequently asked questions", "questions"]
        }
        anchors = anchors_map.get(intent, [])

        # 1) Try snippet-based extraction near anchors
        snips = _collect_snippets(content, anchors)
        snippet_text = "\n".join(snips) if snips else content[:1200]

        # 2) Try list-like extraction (bullets / inline lists)
        items = _as_list_lines(snippet_text)
        if not items:
            items = _as_list_lines(content)

        # 3) Narrow to service-sounding phrases for services intent
        if intent == "services" and items:
            service_keywords = [
                "service", "services", "solution", "solutions", "consulting",
                "development", "design", "integration", "support", "training",
                "implementation", "modernization", "testing", "qa", "maintenance",
                "iot", "blockchain", "app", "mobile", "ui", "ux"
            ]
            filtered = []
            for i in items:
                low = i.lower()
                if any(k in low for k in service_keywords):
                    filtered.append(i)
            if filtered:
                items = filtered

        # 4) Paragraph fallback (sentences containing anchors)
        if intent == "services" and not items:
            sentences = _sentences(content)
            candidates = [s for s in sentences if any(a in s.lower() for a in anchors)]
            if candidates:
                items = [s.strip() for s in candidates[:12]]

        # 5) Regex-based extraction (strong fix)
        if intent == "services":
            regex_candidates = []
            pattern = re.compile(
                r"([A-Za-z0-9&\-/]{2,60}(?:\s+[A-Za-z0-9&\-/]{1,20}){0,6}\s+(?:Service|Services|Solution|Solutions|Development|Design|Support|Consulting|Integration|Modernization|Maintenance|Testing|QA|Applications|Implementation))",
                flags=re.IGNORECASE
            )
            for m in pattern.finditer(content):
                part = m.group(0).strip()
                part = re.sub(r"[,\.;:]+$", "", part).strip()
                regex_candidates.append(part)

            pattern2 = re.compile(
                r"([A-Za-z0-9&\-/ ]{2,60}?(?:app|mobile|iot|blockchain|ui|ux|legacy|software|it)\s+(?:development|services|solutions|applications))",
                flags=re.IGNORECASE
            )
            for m in pattern2.finditer(content):
                part = m.group(0).strip()
                part = re.sub(r"[,\.;:]+$", "", part).strip()
                regex_candidates.append(part)

            normalized = []
            for rc in regex_candidates:
                rc2 = re.sub(r"\s{2,}", " ", rc).strip()
                rc2 = re.sub(r"\bservices\b", "services", rc2, flags=re.I)
                rc2 = re.sub(r"\bsolutions\b", "solutions", rc2, flags=re.I)
                if len(rc2.split()) <= 12 and len(rc2) > 3:
                    normalized.append(rc2)

            if normalized:
                seen = set()
                ordered = []
                for x in normalized + (items or []):
                    k = x.lower()
                    if k not in seen:
                        seen.add(k)
                        ordered.append(x)
                items = ordered

        # 6) Final cleaning + short form generation
        if items:
            cleaned = []
            seen = set()
            for it in items:
                it2 = re.sub(r"^\s*(we\s+|our\s+|with\s+|the\s+)", "", it, flags=re.I).strip()
                it2 = re.sub(r"(?:that|which|that help|that enable).*$", "", it2, flags=re.I).strip()
                it2 = re.sub(r"[^\w\s&\-/()]+$", "", it2).strip()
                if 2 <= len(it2) <= 120:
                    key = it2.lower()
                    if key not in seen:
                        seen.add(key)
                        cleaned.append(it2)

            if cleaned:
                long_ans = "• " + "\n• ".join(cleaned[:12])
                short_ans = ", ".join(cleaned[:6])
                return short_ans, long_ans

        return "", ""

    def _answer_from_content(self, question: str, content: str) -> Tuple[str, str]:
        if not content:
            return "", ""

        # Always force list extraction for service-related queries
        if "service" in question.lower():
            s_short, s_long = self._extract_list_answer(question, content)
            if s_long:
                return s_short, s_long

        # Normal list extraction for other intents
        s_short, s_long = self._extract_list_answer(question, content)
        if s_long:
            return s_short, s_long

        # Extractive QA fallback
        best_span, score = self._extractive_qa_best(question, content)
        if best_span:
            low = content.lower()
            idx = low.find(best_span.lower())
            if idx != -1:
                snip = _expand_to_sentence_bounds(content, max(0, idx - 180), idx + len(best_span) + 180)
                sentences = _sentences(snip)
                long_ans = " ".join(sentences[:3]).strip() if sentences else snip
                short_ans = best_span.strip()
                if len(short_ans) < self.MIN_ANSWER_LENGTH and sentences:
                    short_ans = sentences[0][:180].rstrip() + ("…" if len(sentences[0]) > 180 else "")
                return short_ans, long_ans

        # Token/snippet fallback
        tokens = _tokenize(question, self.STOP)
        snips = _collect_snippets(content, tokens)
        if snips:
            long_ans = "\n\n".join(snips[:2]).strip()
            short_ans = snips[0][:180].rstrip() + ("…" if len(snips[0]) > 180 else "")
            return short_ans, long_ans

        # Generative fallback
        if self.generative_tokenizer and self.generative_qa:
            try:
                safe_ctx = self.generative_tokenizer.decode(
                    self.generative_tokenizer.encode(content, truncation=True, max_length=520),
                    skip_special_tokens=True,
                )
                prompt = ("Answer ONLY using the context below. If the answer is not in the context, "
                          "say 'Not available in the provided content.'\n\n"
                          f"Context:\n{safe_ctx}\n\nQuestion: {question}\n\nAnswer:")
                gen = self.generative_qa(prompt, max_new_tokens=220, do_sample=False)
                long = (gen[0]["generated_text"] if isinstance(gen, list) and gen else str(gen)).replace(prompt, "").strip()
            except Exception as e:
                long = f"Not available in the provided content. ({e})"
            if not long or "not available" in long.lower():
                return "", ""
            short = long.split(".")[0]
            if len(short) < self.MIN_ANSWER_LENGTH:
                short = long[:160].rstrip() + ("…" if len(long) > 160 else "")
            return short.strip(), long.strip()

        return "", ""

    # ---------- Mongo domain-first ----------
    def _mongo_domain_search(self, question: str, root_url: Optional[str]) -> Optional[Dict]:
        if self.scraped_data_collection is None:
            return None

        tokens = _tokenize(question, self.STOP)
        cats = self._question_categories(question)
        qvec = None
        try:
            if self.embedder:
                qvec = self.embedder.encode([question])[0]
        except Exception:
            qvec = None

        query = {}
        if root_url:
            query["url"] = {"$regex": f"^{re.escape(root_url)}"}

        projection = {
            "content": 1,
            "lemmatized_content": 1,
            "title": 1,
            "url": 1,
            "predicted_category": 1,
            "created_at": 1,   # sort by newest
            "topic": 1         # may exist
        }

        candidates = list(
            self.scraped_data_collection.find(query, projection)
            .sort("created_at", -1)
            .limit(180)
        )

        if not candidates and root_url:
            from urllib.parse import urlparse
            parsed = urlparse(root_url)
            root_domain = parsed.netloc
            candidates = list(
                self.scraped_data_collection.find(
                    {"url": {"$regex": f'https?://{re.escape(root_domain)}'}}, projection
                ).sort("created_at", -1)
                .limit(180)
            )

        if not candidates:
            return None

        # --- Contact queries special handling ---
        if self.is_contact_query(question):
            query_type = self.get_contact_query_type(question)

            contact_candidates, other_candidates = [], []
            for doc in candidates:
                url = (doc.get("url") or "").lower()
                title = (doc.get("title") or "").lower()
                if any(term in url or term in title for term in ["contact", "contact-us", "about"]):
                    contact_candidates.append(doc)
                else:
                    other_candidates.append(doc)

            search_order = contact_candidates + other_candidates

            best_email = best_phone = best_address = None
            for doc in search_order:
                content = doc.get("content") or doc.get("lemmatized_content") or ""
                info = self.extract_contact_details(content)
                if not best_email and info["email"]:
                    best_email = info["email"]
                if not best_phone and info["phone"]:
                    best_phone = info["phone"]
                if not best_address and info["address"]:
                    best_address = info["address"]
                if best_email and best_phone and best_address:
                    break

            contact_info = {
                "email": best_email or [],
                "phone": best_phone or [],
                "address": best_address
            }
            msg = self.format_contact_response(contact_info, query_type)
            return {
                "title": "Contact Information",
                "url": (contact_candidates[0].get("url", "") if contact_candidates else root_url) or "",
                "short_answer": msg,
                "long_answer": msg,
                "source": "contact_extraction",
                "quality_score": 1.0,
            }

        # Non-contact logic: score all candidates
        scored = []
        for d in candidates:
            sc = (
                self._score_doc(qvec, tokens, cats, d)
                if qvec is not None
                else self._score_doc([0], tokens, cats, d)
            )
            # Light KeyBERT assist if doc.topic is missing and extractor is available
            if self.topic_extractor and not d.get("topic"):
                try:
                    txt = (d.get("content") or d.get("lemmatized_content") or "")[:1600]
                    topic = self.topic_extractor.extract_main_topic(txt)  # use doc text

                    # topics may be list[str]; add tiny boost for overlap
                    topic_bonus = self._topic_overlap_boost(tokens, topic)
                    sc += topic_bonus
                except Exception:
                    pass

            scored.append((sc, d))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_score, top_doc = (scored[0] if scored else (0.0, None))
        if top_doc is None or top_score < 0.22:
            return None

        content = top_doc.get("content") or top_doc.get("lemmatized_content") or ""
        short_ans, long_ans = self._answer_from_content(question, content)

        if not long_ans:
            sents = _sentences(content)
            long_ans = " ".join(sents[:3])[:600]
            if not short_ans:
                short_ans = (
                    sents[0][:180] + ("…" if len(sents[0]) > 180 else "")
                    if sents
                    else ""
                )

        return {
            "title": top_doc.get("title", "Untitled"),
            "url": top_doc.get("url", ""),
            "short_answer": (short_ans or "").strip(),
            "long_answer": (long_ans or "").strip(),
            "source": "mongo_domain_search",
            "quality_score": float(top_score),
        }

        # ---------- Milvus search ----------
    def semantic_search(self, user_question: str, top_k: int = 5, score_threshold: float = 0.45):
        """
        Perform semantic search in Milvus for the given question.
        Requires 'chunk_id' field to exist in the collection schema.
        """
        if self.collection is None:
            return [], []
        try:
            self.collection.load()
            if not self.embedder:
                return [], []

            # Encode question into vector
            vector = self.embedder.encode([user_question])[0].tolist()

            results = self.collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k * 4,
                output_fields=["url", "title", "topic", "content", "chunk_id"],  # ✅ chunk_id included
            )

            raw_hits = results[0] if results else []

            # Filter by similarity threshold
            filtered = [h for h in raw_hits if h.distance >= score_threshold]

            # Deduplicate by (url, chunk_id) so same chunk isn’t returned multiple times
            seen = set()
            unique_hits = []
            for h in filtered:
                ent = h.entity
                key = (ent.get("url", ""), ent.get("chunk_id", -1))
                if key not in seen:
                    seen.add(key)
                    unique_hits.append(h)
                if len(unique_hits) >= top_k:
                    break

            return unique_hits, raw_hits

        except Exception as e:
            print(f"[ERROR] Semantic search failed: {e}")
            traceback.print_exc()
            return [], []

    # ---------- QA pairs fallback ----------
    def search_qa_pairs(self, question: str, selected_urls: List[str] = None, top_k: int = 5) -> List[Dict]:
        if self.qa_collection is None:
            return []
        try:
            query = {}
            if selected_urls:
                query["url"] = {"$in": selected_urls}
            qa_docs = list(self.qa_collection.find(query))
            if not qa_docs:
                return []
            all_pairs = []
            for doc in qa_docs:
                for qa in doc.get("qa_pairs", []):
                    all_pairs.append({
                        "question": qa.get("question", ""),
                        "short_answer": qa.get("short_answer", ""),
                        "long_answer": qa.get("long_answer", ""),
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "topic": doc.get("topic", "")
                    })
            if not all_pairs:
                return []
            if not self.embedder:
                return all_pairs[:top_k]
            qvec = self.embedder.encode([question])[0]
            out = []
            for qa in all_pairs:
                if not qa["question"]:
                    continue
                try:
                    avec = self.embedder.encode([qa["question"]])[0]
                    sim = cosine_similarity([qvec], [avec])[0][0]
                    qa["similarity_score"] = sim
                    out.append(qa)
                except Exception:
                    continue
            out.sort(key=lambda x: x["similarity_score"], reverse=True)
            return out[:top_k]
        except Exception as e:
            print(f"[ERROR] QA pairs search failed: {e}")
            traceback.print_exc()
            return []

    # ---------- Public: main answer function ----------
    def answer_question(
        self,
        question: str,
        documents: Optional[List[Dict]] = None,
        root_url: Optional[str] = None,
        user_id: Optional[str] = None,
        selected_urls: Optional[List[str]] = None
    ):
        try:
            # --- Step 1: Contact query special case (IMPROVED) ---
            if self.is_contact_query(question) and root_url:
                query_type = self.get_contact_query_type(question)

                best_email = best_phone = best_address = None
                source_url = ""

                if self.scraped_data_collection is not None:
                    query = {"url": {"$regex": f"^{re.escape(root_url)}"}}
                    docs = list(
                        self.scraped_data_collection.find(
                            query,
                            {"content": 1, "title": 1, "url": 1, "created_at": 1}
                        ).sort("created_at", -1)
                    )

                    if not docs and root_url:
                        from urllib.parse import urlparse
                        parsed = urlparse(root_url)
                        root_domain = parsed.netloc
                        docs = list(
                            self.scraped_data_collection.find(
                                {"url": {"$regex": f'https?://{re.escape(root_domain)}'}},
                                {"content": 1, "title": 1, "url": 1, "created_at": 1}
                            ).sort("created_at", -1)
                        )

                    contact_docs, other_docs = [], []
                    for doc in docs:
                        url = (doc.get("url") or "").lower()
                        title = (doc.get("title") or "").lower()
                        if any(term in url or term in title for term in ["contact", "contact-us", "about"]):
                            contact_docs.append(doc)
                        else:
                            other_docs.append(doc)

                    search_docs = contact_docs + other_docs

                    for d in search_docs:
                        info = self.extract_contact_details(d.get("content", ""))
                        if not best_email and info["email"]:
                            best_email = info["email"]
                            source_url = d.get("url", "") or source_url
                        if not best_phone and info["phone"]:
                            best_phone = info["phone"]
                            if not source_url:
                                source_url = d.get("url", "")
                        if not best_address and info["address"]:
                            best_address = info["address"]
                            if not source_url:
                                source_url = d.get("url", "")
                        if best_email and best_phone and best_address:
                            break

                contact_info = {
                    "email": best_email or [],
                    "phone": best_phone or [],
                    "address": best_address
                }
                msg = self.format_contact_response(contact_info, query_type)

                return {
                    "success": True,
                    "answers": [{
                        "title": "Contact Information",
                        "url": source_url or root_url or "",
                        "short_answer": msg,
                        "long_answer": msg,
                        "source": "contact_extraction",
                        "quality_score": 1.0
                    }],
                    "extra_results": [],
                    "question": question
                }

            # --- Step 2: Domain best match ---
            dom_best = self._mongo_domain_search(question, root_url)
            if dom_best:
                return {"success": True, "answers": [dom_best], "extra_results": [], "question": question}

            answers, best_answer = [], None

            # --- Step 3: Extractive QA from provided documents ---
            if documents and self.extractive_qa:
                for doc in documents:
                    ctx = doc.get("content", "")
                    if not ctx.strip():
                        continue
                    try:
                        qa_result = self.extractive_qa(question=question, context=ctx)
                        short = qa_result.get("answer", "").strip()
                    except Exception as e:
                        print(f"[WARNING] Extractive QA failed: {e}")
                        short = ""

                    if not short:
                        sents = _sentences(ctx)
                        short = sents[0][:180] + ("…" if len(sents[0]) > 180 else "") if sents else ""

                    long = self._expand_answer_in_context(short, ctx)

                    if self.embedder:
                        try:
                            qvec = self.embedder.encode([question])[0]
                            avec = self.embedder.encode([long or short])[0]
                            score = cosine_similarity([qvec], [avec])[0][0]
                        except Exception:
                            score = 0.5
                    else:
                        score = 0.5

                    # Topic hint bonus if available
                    if self.topic_extractor:
                        try:
                            topics = self.topic_extractor.extract_topics(ctx[:1600], top_n=5) or []
                            score += self._topic_overlap_boost(_tokenize(question, self.STOP), topics)
                        except Exception:
                            pass

                    ans = {
                        "title": doc.get("title", "Untitled"),
                        "url": doc.get("url", ""),
                        "short_answer": short,
                        "long_answer": long,
                        "source": "extractive_qa",
                        "quality_score": float(score),
                    }
                    answers.append(ans)
                    if not best_answer or score > best_answer["quality_score"]:
                        best_answer = ans

            # --- Step 4: Semantic fallback (Milvus) ---
            if not best_answer and self.collection is not None:
                try:
                    filtered, _ = self.semantic_search(question, top_k=5)
                    seen_urls = set()
                    q_tokens = _tokenize(question, self.STOP)
                    for h in filtered:
                        ent = h.entity
                        url = ent.get("url", "")
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                        ctx = ent.get("content", "")
                        sents = _sentences(ctx)
                        short = (sents[0][:160] + "…") if sents else ""
                        long = " ".join(sents[:3])[:600]

                        base_score = max(0.6, 1.0 - float(h.distance))
                        # Topic bonus from vector record if present
                        base_score += self._topic_overlap_boost(q_tokens, ent.get("topic"))

                        ans = {
                            "title": ent.get("title", "Untitled"),
                            "url": url,
                            "short_answer": short,
                            "long_answer": long,
                            "source": "semantic_search",
                            "quality_score": float(base_score),
                            "similarity": 1.0 - float(h.distance)
                        }
                        answers.append(ans)
                        if not best_answer or ans["quality_score"] > best_answer["quality_score"]:
                            best_answer = ans
                except Exception as e:
                    print(f"[WARNING] Semantic search failed: {e}")

            # --- Step 5: QA pairs fallback ---
            if not best_answer:
                try:
                    qa_pairs = self.search_qa_pairs(question, selected_urls, top_k=3)
                    for qa in qa_pairs:
                        ans = {
                            "title": qa.get("title", "Untitled"),
                            "url": qa.get("url", ""),
                            "short_answer": qa.get("short_answer", "") or qa.get("long_answer", "")[:160] + "…",
                            "long_answer": qa.get("long_answer", ""),
                            "source": "qa_pairs",
                            "quality_score": float(qa.get("similarity_score", 0.0)),
                        }
                        # Topic nudge
                        ans["quality_score"] += self._topic_overlap_boost(_tokenize(question, self.STOP), qa.get("topic"))
                        answers.append(ans)
                        if not best_answer or ans["quality_score"] > best_answer["quality_score"]:
                            best_answer = ans
                except Exception as e:
                    print(f"[WARNING] QA pairs search failed: {e}")

            # --- Step 6: Handle list-type questions ---
            if answers and any(kw in (question or "").lower() for kw in ["list", "names", "who are", "what are", "services"]):
                if "service" in (question or "").lower() and best_answer:
                    return {
                        "success": True,
                        "answers": [best_answer],
                        "extra_results": [a for a in answers if a is not best_answer],
                        "question": question
                    }
                valid_answers = [a for a in answers if (a.get("short_answer") or "").strip()]
                if valid_answers:
                    merged = "\n- " + "\n- ".join([a["short_answer"] for a in valid_answers])
                    return {
                        "success": True,
                        "answers": [{
                            "short_answer": merged,
                            "long_answer": merged,
                            "url": valid_answers[0]["url"],
                            "title": valid_answers[0]["title"],
                            "source": "list_merge",
                            "quality_score": 1.0
                        }],
                        "extra_results": answers,
                        "question": question
                    }

            # --- Step 7: Final return ---
            if best_answer and best_answer["quality_score"] >= self.MIN_SIMILARITY_THRESHOLD:
                answers = [best_answer] + [a for a in answers if a is not best_answer]
                return {"success": True, "answers": answers, "extra_results": [], "question": question}

            # --- Step 8: Topic-only fallback when absolutely nothing scored well ---
            if not answers and self.topic_extractor and root_url and self.scraped_data_collection is not None:
                # Try topic-guided snippet from latest docs for the domain
                try:
                    docs = list(
                        self.scraped_data_collection.find(
                            {"url": {"$regex": f"^{re.escape(root_url)}"}},
                            {"content": 1, "title": 1, "url": 1, "created_at": 1}
                        ).sort("created_at", -1).limit(10)
                    )
                    q_tokens = _tokenize(question, self.STOP)
                    for d in docs:
                        txt = (d.get("content") or "")[:2000]
                        topics = self.topic_extractor.extract_topics(txt, top_n=5) or []
                        if self._topic_overlap_boost(q_tokens, topics) > 0:
                            # Take first good snippet
                            snips = _collect_snippets(txt, topics)
                            if snips:
                                short = snips[0][:180] + ("…" if len(snips[0]) > 180 else "")
                                long = "\n\n".join(snips[:2])
                                return {
                                    "success": True,
                                    "answers": [{
                                        "title": d.get("title", "Untitled"),
                                        "url": d.get("url", ""),
                                        "short_answer": short,
                                        "long_answer": long,
                                        "source": "topic_snippet_fallback",
                                        "quality_score": 0.35
                                    }],
                                    "extra_results": [],
                                    "question": question
                                }
                except Exception:
                    pass

            # --- Step 9: Partial results or failure ---
            if answers:
                return {
                    "success": False,
                    "answers": [],
                    "extra_results": answers,
                    "question": question,
                    "message": "I found some related information but couldn't find a definitive answer."
                }

            return {
                "success": False,
                "answers": [],
                "extra_results": [],
                "question": question,
                "message": "I couldn't find a relevant answer in the selected content."
            }

        except Exception as e:
            print(f"[ERROR] Question answering failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "answers": [],
                "extra_results": [],
                "question": question
            }

    def _expand_answer_in_context(self, answer: str, context: str) -> str:
        """Finds the full sentence containing the answer for better readability."""
        if not answer or not context:
            return answer
        sentences = _sentences(context)
        for s in sentences:
            if answer in s:
                return s.strip()
        return answer

    # ---------- Back-compat helpers ----------
    def extract_answers_with_confidence(self, question: str, context: str) -> Tuple[str, str, float]:
        short, long = self._answer_from_content(question, context)
        conf = 0.0
        try:
            if self.embedder and (long or short):
                qvec = self.embedder.encode([question])[0]
                avec = self.embedder.encode([long or short or ""])[0]
                conf = float(cosine_similarity([qvec], [avec])[0][0])
        except Exception:
            pass
        return (short or ""), (long or ""), conf

    def extract_answers(self, question, context):
        short, long, _ = self.extract_answers_with_confidence(question, context)
        return short, long

    # ---------- No-op hooks ----------
    def refresh_index(self):
        try:
            if self.collection is not None:
                self.collection.load()
        except Exception:
            pass

    def close_connection(self):
        try:
            if self.mongo_client is not None:
                self.mongo_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    qa = QuestionAnswerer()
    print("Ready.")
