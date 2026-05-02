import logging
from pathlib import Path

import torch
import yaml
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

# Imports modernos do LangChain v0.3+
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Carrega configurações centralizadas do YAML."""
    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """Pipeline RAG (Retrieval-Augmented Generation) Otimizado.

    Mantém o modelo de embeddings e a conexão do banco em memória para baixa latência.
    """

    def __init__(self) -> None:
        self.cfg = load_config()
        self.rag_cfg = self.cfg.get("rag", {})

        # Configuração de Hardware Otimizada
        self.device = (
            "xpu"
            if hasattr(torch, "xpu") and torch.xpu.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Inicializando Embeddings no device: {self.device}")

        # Carregamento Único do Modelo de Embeddings
        model_name = self.rag_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Inicialização e Conexão com o ChromaDB
        self.db_path = self.rag_cfg.get("vector_db_path", "data/chroma_db")
        self.docs_dir = self.rag_cfg.get("docs_dir", "data/documents")

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma(
            collection_name="base_conhecimento_empresa",
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )

    def ingest_directory(self) -> None:
        """Lê todos os PDFs e TXTs e injeta no ChromaDB de forma incremental.

        Evita duplicatas básicas verificando se o banco já possui dados antes da ingestão em lote.
        """
        docs_dir = Path(self.docs_dir)
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"Diretório de documentos criado, mas está vazio: {docs_dir}"
            )
            return

        # Verificação de Ingestão Incremental
        # Se o banco já tem documentos, evitamos re-processar tudo (estratégia simples)
        try:
            count = self.vector_store._collection.count()
            if count > 0:
                logger.info(
                    f"O banco vetorial já possui {count} documentos. "
                    "Pulando ingestão completa para evitar duplicatas."
                )
                return
        except Exception as e:
            logger.warning(f"Não foi possível verificar contagem do banco: {e}")

        logger.info(f"Iniciando ingestão de novos documentos em {docs_dir}...")

        # Carregadores em lote
        txt_loader = DirectoryLoader(
            str(docs_dir), glob="**/*.txt", loader_cls=TextLoader
        )
        pdf_loader = DirectoryLoader(
            str(docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader
        )

        raw_documents = txt_loader.load() + pdf_loader.load()

        if not raw_documents:
            logger.warning("Nenhum arquivo .txt ou .pdf encontrado para ingestão.")
            return

        # Chunking Otimizado
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_cfg.get("chunk_size", 1000),
            chunk_overlap=self.rag_cfg.get("chunk_overlap", 200),
            separators=["\n\n", "\n", ".", " ", ""],
        )

        chunks = splitter.split_documents(raw_documents)

        # Inserção no Banco Vetorial
        self.vector_store.add_documents(documents=chunks)
        logger.info(
            f"Ingestão concluída: {len(chunks)} fragmentos adicionados ao ChromaDB."
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Busca os contextos mais relevantes no banco para a pergunta atual."""
        if not query or not query.strip():
            logger.warning("Query vazia recebida no RAGPipeline.")
            return []

        top_k = top_k or self.rag_cfg.get("top_k", 3)
        logger.debug(f"Buscando contexto para: '{query}' (top_k={top_k})")

        try:
            return self.vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error(f"Erro ao recuperar contextos do ChromaDB: {e}")
            return []


if __name__ == "__main__":
    # Script auxiliar rodado diretamente no terminal para popular o banco
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("--- Ingestão de Documentos (RAG) ---")
    pipeline = RAGPipeline()
    pipeline.ingest_directory()
