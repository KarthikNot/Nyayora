import os

# ========================= IMPORTANT CONFIGS =========================
IPC_DATASET_DOWNLOAD_LINK: str = "https://www.indiacode.nic.in/repealedfileopen?rfilename=A1860-45.pdf"
BNS_DATASET_2023_LINK: str = "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf"
BNSS_DATASET_2023_LINK: str = "https://www.indiacode.nic.in/bitstream/123456789/21544/1/the_bharatiya_nagarik_suraksha_sanhita%2C_2023.pdf"
BSA_DATASET_2023_LINK: str = "https://www.indiacode.nic.in/bitstream/123456789/20063/1/aa202347.pdf?utm_source=chatgpt.com"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# ========================= FILE PATHS =========================
BASE_PATH: str = os.getcwd()
DATASET_PATH: str = os.path.join(BASE_PATH, "data", "indian_penal_code.pdf") 
PREPROCESSED_DATASET_PATH: str = os.path.join(BASE_PATH, "artifacts", "chunked_dataset.pkl")
VECTOR_EMBEDDINGS_PATH: str = os.path.join(BASE_PATH, "artifacts", "vectore_store.pkl")
CHROMA_DB_PATH: str = os.path.join(BASE_PATH, "artifacts", "chroma_db")

# ========================= CHUNKER CONFIG ============================
CHUNK_SIZE = 700
CHUNK_OVERLAP = 125

# ========================= OLLAMA MODELS CONFIG ======================
VECTOR_EMBEDDINGS_MODEL = "mxbai-embed-large"
EMBEDDING_BATCH_SIZE: int = 500
LLM_MODEL = "gemma4:e4b"
MINI_AGENT_MODELS = "llama3.2:3b"

REASONING_MODEL_TEMP: float = 0.1
MINI_MODEL_TEMP: float = 0.0
DOCS_TO_RETRIEVE: int = 50
DOCS_TO_RERANK: int = 50

# ========================= LLM MODEL PROMPTS =========================
# ================= MULTI QUERY PROMPT =================
MULTI_QUERY_PROMPT = """You are an expert legal assistant. Your task is to generate exactly 10 distinct, high-quality search variations of the user's query. 
Focus on capturing colloquial language, different phrasings, synonyms for crimes, or specific legal terms an ordinary Indian citizen might use. 
Ensure the variations look at the query from multiple perspectives to maximize distance-based vector database retrieval.

Original question: {question}"""


# ================= PLANNER PROMPT =====================
PLANNER_AGENT_PROMPT: str = """You are a legal query strategist. Break down the user's complex legal query into 3-4 distinct sub-questions.
Focus on: 1. Legal Definitions, 2. Procedural Requirements (Arrest/Bail), 3. Punishments, and 4. Specific IPC Section lookups.

User Query: {query}"""


# ================= CLASSIFIER PROMPT ==================
CLASSIFIER_AGENT_PROMPT: str = """You are a highly precise binary classifier for a legal AI system. Your sole task is to determine if a given user query is related to criminal law or legal procedures.

Criteria for TRUE (is_legal = true):
- The query mentions criminal offenses (theft, assault, fraud, extortion, harassment, etc.), even if a specific country is not mentioned.
- The query references Indian law, IPC/BNS/BNSS/BSA sections, police protocols, FIRs, bail, or court proceedings.
- General legal questions about rights, penalties, or procedures.
- Queries involving online content removal, privacy violations, defamation, intellectual property disputes, or unauthorized use of personal media.

Criteria for FALSE (is_legal = false):
- Explicit mentions of non-Indian jurisdictions (e.g., "In the US...", "under UK law", "California laws").
- General conversation, greetings, historical facts unrelated to criminal legal matters, jokes, tech support, or food.

Examples (ensure confidence_score is a float between 0 and 1):
- "Someone stole my car keys" -> TRUE (General crime)
- "What is IPC 420?" -> TRUE
- "Can police enter my house without a warrant?" -> TRUE
- "Someone uploaded my video without my permission and is demanding money to remove it." -> TRUE (Extortion/Privacy)
- "Can I be arrested for this in New York?" -> FALSE (Non-Indian jurisdiction)
- "How do I cook biryani?" -> FALSE
- "Who is the current prime minister?" -> FALSE"""


# ================= INTENT PROMPT ======================
INTENT_AGENT_PROMPT: str = """You are a legal intent classifier. Your task is to categorize the primary nature of the legal incident described in the query into exactly one of the designated classes.

Available Categories:
- fraud (cheating, fake checks, forgery, scams)
- theft (stolen property, burglary, robbery)
- assault (physical attacks, battery, fights)
- murder (homicide, killing, fatal injuries)
- cybercrime (hacking, online scams, data theft)
- extortion (blackmail, demanding money under threat)
- harassment (stalking, abusive messages, bullying)
- property_dispute (land grabbing, rental issues, illegal occupation)
- legal_advice (abstract questions about generic IPC sections or legal definitions)
- other (any legal matter that does not fit the descriptions above)

Examples (ensure confidence_score is a float between 0 and 1):
- "My neighbor moved his fence onto my land." -> property_dispute
- "Someone took my phone from my bag while I was sleeping." -> theft
- "I was forced to pay money because they threatened to leak my photos." -> extortion
- "What is the meaning of Culpable Homicide?" -> legal_advice
- "Someone broke into my car and stole my keys." -> theft

Select the single category that best represents the user's situation based on the definitions above.

STRICT INSTRUCTIONS FOR STATUTES:
1. If the query is about a crime (theft, murder, fraud, etc.), ALWAYS include BOTH ["ipc", "bns"] in 'target_statutes'.
2. If it is about police procedure or arrest, include ["bnss", "ipc"].
3. If it is about evidence or court documents, include ["bsa", "ipc"].
4. Only use "judgments" if specifically asked for case law."""


# ================= LAW AGENT PROMPT ===================
LAW_AGENT_PROMPT: str = """You are Nyayora, an expert AI legal assistant specializing in Indian criminal law. Your objective is to populate the requested JSON schema accurately using ONLY the provided legal context. 
Even if a user query is phrased generally (e.g., 'someone stole my car keys'), you must interpret and answer it strictly according to the Indian statutes (IPC, BNS, etc.) found in the context.

STRICT OPERATIONAL DIRECTIVES:
1. Grounding: Rely exclusively on the provided legal context. Do NOT extrapolate or assume sections that are not explicitly found in the retrieved chunks.
2. Legal Specifics: For every section found, you MUST identify if it is Bailable, Cognizable, and which Court handles it based on the text.
3. Missing Sections: If the user queries a section not visible in your context, leave the 'sections' array empty and use the 'limitations' field to explain.
4. Depth of Analysis: Provide a thorough, detailed 'interpretation' of each section. Explain the legal requirements (e.g., intent, act, harm) based on the text. The 'summary' should be a comprehensive legal overview, not just a brief snippet.
5. Language & Tone: Maintain a highly formal, professional, and objective tone. Use precise legal terminology where appropriate while ensuring readability.
6. Urgency: Use 'urgency_level' (Low, Medium, High) to indicate if immediate legal intervention is required.
7. Status Enums: For bailable/cognizable/is_compoundable fields, strictly use: "Bailable", "Non-Bailable", "Cognizable", "Non-Cognizable", "Compoundable", "Non-Compoundable", or "Not Specified".
8. Professional Boundaries: Do not provide prescriptive personal directives; frame steps as generic options. Always complete the designated schema layout."""


# ========================= EVALUATION CONFIG =========================
TEST_QUESTIONS = [
    "What is the punishment for theft under IPC?",
    "Define murder and its punishment according to the Indian Penal Code.",
    "What constitutes cheating in Indian law?",
    "What is the penalty for extortion under IPC?",
    "What is robbery under IPC?",
    "What is the punishment for robbery?",
    "What is criminal breach of trust?",
    "What is the punishment for criminal breach of trust?",
    "What is grievous hurt under IPC?",
    "What is wrongful restraint under IPC?",
    "What is wrongful confinement under IPC?",
    "What is criminal intimidation?",
    "What is the punishment for criminal intimidation?",
    "What is house trespass under IPC?",
    "What is lurking house trespass?",
    "What is mischief under IPC?",
    "What is defamation under IPC?",
    "What is forgery under IPC?",
    "What is the punishment for forgery?",
    "What is rioting under IPC?",
    "What is unlawful assembly under IPC?"
]


GROUND_TRUTHS = [
    "Section 378 defines theft. Section 379 prescribes punishment for theft with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
    "Section 300 defines murder. Section 302 prescribes punishment for murder with death, or imprisonment for life, and also fine.",
    "Section 415 defines cheating as deceiving a person and dishonestly or fraudulently inducing that person to deliver property, retain property, or act in a way that causes damage or harm.",
    "Section 383 defines extortion. Section 384 prescribes punishment for extortion with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
    "Section 390 defines robbery as theft or extortion accompanied by violence, threat of instant death, hurt, or wrongful restraint.",
    "Section 392 prescribes punishment for robbery with rigorous imprisonment for a term which may extend to ten years, and also fine.",
    "Section 405 defines criminal breach of trust as dishonest misappropriation, conversion, use, or disposal of property entrusted to a person.",
    "Section 406 prescribes punishment for criminal breach of trust with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
    "Section 320 defines grievous hurt and enumerates specific serious bodily injuries recognized under law.",
    "Section 339 defines wrongful restraint as voluntarily obstructing a person so as to prevent that person from proceeding in a direction in which they have a right to proceed.",
    "Section 340 defines wrongful confinement as wrongfully restraining a person in such a manner as to prevent that person from proceeding beyond certain circumscribed limits.",
    "Section 503 defines criminal intimidation as threatening another with injury to person, reputation, or property with intent to cause alarm.",
    "Section 506 prescribes punishment for criminal intimidation with imprisonment, fine, or both, depending on the nature of the threat.",
    "Section 442 defines house-trespass as criminal trespass into a building, tent, or vessel used as a human dwelling, place of worship, or place for custody of property.",
    "Section 443 defines lurking house-trespass as house-trespass where the offender takes precautions to conceal the trespass from a person entitled to exclude or eject the trespasser.",
    "Section 425 defines mischief as causing destruction of property or a change that diminishes its value or utility, with intent or knowledge of causing wrongful loss or damage.",
    "Section 499 defines defamation as making or publishing imputations concerning any person with intent, knowledge, or reason to believe that such imputation will harm that person's reputation.",
    "Section 463 defines forgery as making a false document or electronic record with intent to cause damage, support a claim, commit fraud, or facilitate cheating.",
    "Section 465 prescribes punishment for forgery with imprisonment of either description for a term which may extend to two years, or with fine, or with both.",
    "Section 146 defines rioting as the use of force or violence by an unlawful assembly or by any member thereof in prosecution of the common object of such assembly.",
    "Section 141 defines unlawful assembly as an assembly of five or more persons with a common object specified by law."
]