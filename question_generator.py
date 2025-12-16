# question_generator.py
import torch
from transformers import pipeline
from question.question_answerer import QuestionAnswerer
from pymongo import MongoClient
import traceback
from typing import Optional


class QuestionGenerator:
    def __init__(self,
                 model_name: str = "google/flan-t5-base",   # ⚠️ smaller model for stability
                 mongo_uri: Optional[str] = None,
                 qa_tool: Optional[QuestionAnswerer] = None):
        """
        QuestionGenerator handles generating questions from content and answering them
        using a QuestionAnswerer + HuggingFace pipeline.

        Parameters
        ----------
        model_name : str
            HF model name for question generation.
        mongo_uri : str | None
            MongoDB URI. Defaults to mongodb://localhost:27017/
        qa_tool : QuestionAnswerer | None
            Optional injected QA tool instance. If None, will create one internally.
        """
        device = -1  # use CPU by default

        # --- Init Question Generation pipeline ---
        self.q_gen = None
        try:
            print(f"[INFO] Initializing question-generation pipeline: {model_name} (device={device})")
            self.q_gen = pipeline(
                "text2text-generation",
                model=model_name,
                device=device
            )
            print("[✓] Question-generation pipeline initialized")
        except Exception as e:
            print(f"[WARNING] Failed to initialize question-generation pipeline '{model_name}': {e}")
            traceback.print_exc()
            self.q_gen = None

        # --- Init QuestionAnswerer (match process_chat style) ---
        if qa_tool is not None:
            self.qa_tool = qa_tool
            print("[✓] Using injected QuestionAnswerer instance")
        else:
            try:
                # question_generator.py
                self.qa_tool = QuestionAnswerer(collection_name="webpages")

                print("[✓] QuestionAnswerer initialized")
            except Exception as e:
                print(f"[✗] Failed to initialize QuestionAnswerer: {e}")
                traceback.print_exc()
                self.qa_tool = None

        # --- Init MongoDB connection (optional) ---
        try:
            uri = mongo_uri or "mongodb://localhost:27017/"
            self.mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client['chatbotDB']
            self.scraped_data_collection = self.db['scraped_data']
            # test connection
            self.mongo_client.server_info()
            print("[✓] Connected to MongoDB")
        except Exception as e:
            print(f"[WARNING] MongoDB connection failed: {e}")
            traceback.print_exc()
            self.mongo_client = None
            self.db = None
            self.scraped_data_collection = None

    # ----------------------------------------------------------------------

    def _draft_questions(self, topic, category, content, min_q=3, max_q=5):
        """
        Use the text2text pipeline to draft candidate questions.
        """
        if not self.q_gen:
            print("[WARNING] Question-generation pipeline not available")
            return []

        questions = set()
        attempts = 0
        safe_context = content[:2000]  # avoid overly long input

        while len(questions) < min_q and attempts < 5:
            prompt = (
                f"Generate {max_q} distinct, informative questions about the topic "
                f"'{topic}' (category: '{category}'). Return one question per line.\n\n"
                f"Context:\n{safe_context}"
            )
            try:
                res = self.q_gen(
                    prompt,
                    max_new_tokens=256,  # ✅ only use this
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                    top_p=0.95,
                    truncation=True     # ✅ prevent sequence too long warnings
                )

                text_out = ""
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                    text_out = res[0].get("generated_text", "")
                elif isinstance(res, str):
                    text_out = res
                else:
                    text_out = str(res)

                for q in text_out.split("\n"):
                    q = q.strip()
                    if q and len(q) > 10:
                        questions.add(q)
            except Exception as e:
                print(f"[WARNING] Question generation attempt {attempts + 1} failed: {e}")
                traceback.print_exc()
            attempts += 1

        return list(questions)[:max_q]

    # ----------------------------------------------------------------------

    def generate_questions_and_answers(self, topic, category, content, min_q=3, max_q=5):
        """
        Generate QA pairs: generate questions, then answer them with QA tool.
        """
        questions = self._draft_questions(topic, category, content, min_q=min_q, max_q=max_q)
        qa_pairs = []

        if not questions:
            questions = [f"What is this page about?"]

        for q in questions:
            try:
                if self.qa_tool:
                    short_ans, long_ans = self.qa_tool.extract_answers(q, content)
                else:
                    short_ans = f"This page is about {topic}."
                    long_ans = f"This page discusses {topic} in the context of {category}."
                qa_pairs.append({
                    "question": q,
                    "short_answer": short_ans,
                    "long_answer": long_ans,
                })
            except Exception as e:
                print(f"[WARNING] Failed to generate answer for question: {q} - {e}")
                traceback.print_exc()
                qa_pairs.append({
                    "question": q,
                    "short_answer": f"Information about {topic}",
                    "long_answer": f"This content discusses {topic} in the context of {category}.",
                })

        return qa_pairs

    # ----------------------------------------------------------------------

    def save_all_to_mongodb(self):
        """Generate QA pairs from all scraped data in MongoDB."""
        if not self.mongo_client or not self.scraped_data_collection:
            print("[ERROR] MongoDB not connected - cannot generate QA pairs from DB")
            return

        try:
            documents = list(self.scraped_data_collection.find({}))
            if not documents:
                print("[WARNING] No scraped data found in MongoDB")
                return

            print(f"[INFO] Found {len(documents)} documents in MongoDB")

            for doc in documents:
                try:
                    topic = doc.get("topic", "Unknown")
                    category = doc.get("predicted_category", "General")
                    content = doc.get("content", "")
                    url = doc.get("url", "")
                    title = doc.get("title", "")

                    if not content.strip():
                        continue

                    print(f"🔄 Generating QA for: {title[:50]}... - {topic}/{category}")
                    qa_pairs = self.generate_questions_and_answers(topic, category, content)

                    if qa_pairs:
                        qa_doc = {
                            'url': url,
                            'title': title,
                            'topic': topic,
                            'category': category,
                            'qa_pairs': qa_pairs,
                            'user_id': doc.get('user_id'),
                            'total_pairs': len(qa_pairs)
                        }
                        self.db['qa_pairs'].update_one(
                            {'url': url, 'user_id': doc.get('user_id')},
                            {'$set': qa_doc},
                            upsert=True
                        )
                        print(f"✅ Generated {len(qa_pairs)} QA pairs for {title}")
                except Exception as e:
                    print(f"[ERROR] Failed to process document {doc.get('_id')}: {e}")
                    traceback.print_exc()
                    continue

            print("✅ QA pair generation completed")

        except Exception as e:
            print(f"[ERROR] Failed to generate QA pairs from MongoDB: {e}")
            traceback.print_exc()

    # ----------------------------------------------------------------------

    def save_all_to_file(self, input_path=None, output_path=None):
        """Deprecated: use save_all_to_mongodb instead."""
        print("[INFO] Redirecting to MongoDB-based QA generation...")
        self.save_all_to_mongodb()
