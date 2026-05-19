"""
rag_pipeline.py — Core RAG logic.
Connects ChromaDB (retrieval) → Groq LLM (generation) via LangChain.
Provides specialised query functions for each of the 4 features.
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


def get_vectorstore():
    """Load the persisted ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def get_llm():
    """Initialise the Groq LLM via LangChain."""
    return ChatGroq(
        model_name=GROQ_MODEL,
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def build_chain(system_prompt: str):
    """Build a RetrievalQA chain with a custom prompt."""
    llm = get_llm()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


# ════════════════════════════════════════════════════════════════════
# Feature 1: Express Draft
# ════════════════════════════════════════════════════════════════════
EXPRESS_DRAFT_PROMPT = """You are a professional Indian legal document drafter.
Using the legal knowledge below and the user's details, generate a complete,
court-ready legal document. Use formal legal language, proper formatting with
numbered clauses, and include all standard sections expected in this type of
document under Indian law.

Legal Knowledge:
{context}

User Request:
{question}

Generate the full legal document now:"""


def query_express_draft(doc_type: str, details: dict, extra_text: str = "") -> dict:
    """Generate a legal document draft."""
    chain = build_chain(EXPRESS_DRAFT_PROMPT)
    detail_str = "\n".join([f"- {k}: {v}" for k, v in details.items() if v])
    question = f"Generate a {doc_type} document with these details:\n{detail_str}"
    if extra_text:
        question += f"\n\nAdditional reference text from uploaded file:\n{extra_text}"

    result = chain.invoke({"query": question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])})
    return {"document": result["result"], "sources": sources}


# ════════════════════════════════════════════════════════════════════
# Feature 2: Clause Checker
# ════════════════════════════════════════════════════════════════════
CLAUSE_CHECKER_PROMPT = """You are an expert Indian legal clause analyst.
Analyse the following legal clause and provide:

1. **Plain English Explanation**: What this clause actually means in simple terms.
2. **Risk Level**: Classify as "Low Risk", "Medium Risk", or "High Risk".
3. **Who It Favors**: Which party benefits from this clause.
4. **Red Flags**: Any potentially harmful or one-sided provisions.
5. **Legal Basis**: Relevant Indian laws or precedents that apply.

Legal Knowledge:
{context}

Clause to Analyse:
{question}

Provide your structured analysis:"""


def query_clause_checker(clause_text: str, extra_text: str = "") -> dict:
    """Analyse a legal clause."""
    chain = build_chain(CLAUSE_CHECKER_PROMPT)
    question = clause_text
    if extra_text:
        question += f"\n\nAdditional text from uploaded document:\n{extra_text}"

    result = chain.invoke({"query": question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])})

    # Try to extract risk level from the answer
    answer = result["result"]
    risk = "Medium Risk"
    if "high risk" in answer.lower():
        risk = "High Risk"
    elif "low risk" in answer.lower():
        risk = "Low Risk"

    return {"answer": answer, "risk_level": risk, "sources": sources}


# ════════════════════════════════════════════════════════════════════
# Feature 3: Case Miner
# ════════════════════════════════════════════════════════════════════
CASE_MINER_PROMPT = """You are an expert Indian legal researcher specialising in
landmark Supreme Court and High Court cases.

Based on the legal knowledge provided and the user's query, find and present
the most relevant Indian court cases. For each case provide:
- Case name and citation
- Court and year
- Key facts (brief)
- Holding / Judgment
- Relevance to the query

If no exact match is found, provide the closest related cases and explain
their relevance.

Legal Knowledge:
{context}

Query:
{question}

Present your findings:"""


def query_case_miner(query: str, extra_text: str = "") -> dict:
    """Search for relevant legal cases."""
    chain = build_chain(CASE_MINER_PROMPT)
    question = query
    if extra_text:
        question += f"\n\nAdditional context from uploaded document:\n{extra_text}"

    result = chain.invoke({"query": question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])})
    return {"answer": result["result"], "sources": sources}


# ════════════════════════════════════════════════════════════════════
# Feature 4: Legal Mind
# ════════════════════════════════════════════════════════════════════
LEGAL_MIND_PROMPT = """You are Legal Mind, an advanced AI legal reasoning engine
specialising in Indian law. Provide detailed, structured legal analysis.

Format your response in these exact sections:
1. Legal Issue: Identify the core legal question
2. Applicable Law: Cite specific Indian statutes, articles, and sections
3. Relevant Precedents: Reference landmark cases
4. Legal Analysis: Step-by-step reasoning
5. Conclusion: Clear, actionable conclusion

Legal Knowledge:
{context}

Question:
{question}

Provide your structured legal analysis:"""


def query_legal_mind(question: str, context_text: str = "", extra_text: str = "") -> dict:
    """Deep legal reasoning and analysis."""
    chain = build_chain(LEGAL_MIND_PROMPT)
    full_question = question
    if context_text:
        full_question += f"\n\nCase Context / Facts:\n{context_text}"
    if extra_text:
        full_question += f"\n\nAdditional text from uploaded document:\n{extra_text}"

    result = chain.invoke({"query": full_question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])})
    return {"answer": result["result"], "sources": sources}
