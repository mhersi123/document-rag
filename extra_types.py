import pydantic

class RAGChunkSource(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGInsertResult(pydantic.BaseModel):
    inngested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int