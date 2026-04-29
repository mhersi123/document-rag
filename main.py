import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_then_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from extra_types import RAGQueryResult, RAGSearchResult, RAGInsertResult, RAGChunkSource

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="document-rag",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest Document",
    trigger=inngest.TriggerEvent(event="rag/ingest_document")
)
async def rag_ingest_document(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkSource:
        doc_path = ctx.event.data["doc_path"]
        source_id = ctx.event.data.get("source_id", doc_path)
        ch = load_then_chunk_pdf(doc_path)
        return RAGChunkSource(chunks=ch, source_id=source_id)

    def _insert(chunks_and_src: RAGChunkSource) -> RAGInsertResult:
        ch = chunks_and_src.chunks
        s = chunks_and_src.source_id
        vectors = embed_texts(ch)

        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{s}: {i}")) for i in range(len(ch))]
        payloads = [{"source": s, "text": ch[i]} for i in range(len(ch))]
        QdrantStorage().insert(ids, vectors, payloads)
        return RAGInsertResult(inngested=len(ch))

    chunks_and_src = await ctx.step.run("load-then-chunk", lambda: _load(ctx), output_type=RAGChunkSource)
    data = await ctx.step.run("embed-then-insert", lambda: _insert(chunks_and_src), output_type=RAGInsertResult)
    return data.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query Document",
    trigger=inngest.TriggerEvent(event="rag/query_document_ai")
)
async def rag_query_doc_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int=5) -> RAGSearchResult:
        query_vector = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vector, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 10))
    found = await ctx.step.run("embed_and_search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    print("Contexts returned:", found.contexts)
    print("Num contexts:", len(found.contexts))
    print("Top K: ", top_k)
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = f"""
        Use the following context to answer the question.
        \n\n
        Context:
        {context_block}
        \n\n
        Question: 
        {question}
        \n\n
        Answer concisely using the context above.
        """

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions based on provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_document, rag_query_doc_ai])