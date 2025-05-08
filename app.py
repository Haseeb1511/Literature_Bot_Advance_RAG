import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever,ContextualCompressionRetriever
from langchain_community.document_transformers import LongContextReorder,EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Novel Chatbot",layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])



path_to_got = "data/got/game of thrones.pdf"
path_to_sh = "data/Sherlock Holmes/cano.pdf"
@st.cache_resource(show_spinner="Loading PDF...")
def pdf_loader(path):
    """Document Loading"""
    loader = PyPDFLoader(path)
    return loader.load()

@st.cache_resource(show_spinner="Creating Chunks...")
def text_splitter(_doc):
    """Text chunking """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=300)
    return splitter.split_documents(_doc)


@st.cache_resource(show_spinner="Loading Embedding model...")
def embedding_model():
    """Embedding model"""
    return GoogleGenerativeAIEmbeddings(model = "models/embedding-001")


@st.cache_resource(show_spinner="Creating Vector Store...")
def vector_store_create(_chunks):
    vector_store = FAISS.from_documents(documents=_chunks,embedding = embedding)    
    return vector_store              

# if "got_doc" not in st.session_state:
#     st.session_state.got_doc = pdf_loader(path_to_got)
# if "wc_doc" not in st.session_state:
#     st.session_state.wc_doc = pdf_loader(path_to_sh)

query = st.chat_input("Enter query")
if query:
    got_doc = pdf_loader(path_to_got)
    wc_doc = pdf_loader(path_to_sh)
    if got_doc and wc_doc:
        got_chunks = text_splitter(got_doc)
        sh_chunks = text_splitter(wc_doc)
        if got_chunks and sh_chunks:
            embedding = embedding_model()
            got_vector_store = vector_store_create(got_chunks)
            sh_vector_store = vector_store_create(sh_chunks)

            if got_vector_store and sh_vector_store:
                got_retriever = got_vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3})
                sh_retriever = sh_vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3})

                merger_retrival = RunnableParallel({
                    "got_ret":got_retriever,
                    "sh_ret":sh_retriever
                }) | RunnableLambda(lambda x: x["got_ret"]+x["sh_ret"])

                bm25_retriever_got = BM25Retriever.from_documents(got_chunks)
                bm25_retriever_got.k=3
                bm25_retriever_sh = BM25Retriever.from_documents(sh_chunks)
                bm25_retriever_sh.k=3

                got_hybrid = EnsembleRetriever(
                    retrievers=[bm25_retriever_got,got_retriever],
                    weights=[0.5,0.5])
                sh_hybrid = EnsembleRetriever(
                    retrievers=[bm25_retriever_sh,sh_retriever],
                    weights=[0.5,0.5])

                retriever_chain = RunnableParallel({
                    "got":got_hybrid,
                    "sh":sh_hybrid
                }) | RunnableLambda(lambda x :x["got"]+x["sh"])

                # reranker using cohere
                reranker = CohereRerank(model="rerank-english-v3.0")  
                filter = EmbeddingsRedundantFilter(embeddings=embedding) # Removes duplicate or highly similar chunks.
                reordering = LongContextReorder()  # Reorders documents to maximize coherence in long context windows
                pipeline = DocumentCompressorPipeline(transformers=[reranker,filter,reordering])

                compression_retriever = ContextualCompressionRetriever(
                    base_retriever=retriever_chain,
                    base_compressor=pipeline)

                def context(document):
                    return "\n\n".join(doc.page_content for doc in document)

                model = ChatGroq(model="gemma2-9b-it",max_tokens=512)

                parser = StrOutputParser()
                prompt = PromptTemplate(
                    template="""
                You are a knowledgeable and articulate literary assistant specializing in fictional worlds.

                Using only the CONTEXT provided below, answer the user's QUESTION with clear and detailed storytelling. 
                If the context does not contain enough information, reply in a graceful and narrative tone like:
                    "The tale holds no record of such an event."

                Keep your answer informative, accurate, and grounded strictly in the given context.

                CONTEXT:
                {context}

                QUESTION:
                {question}

                Your Answer:
                """,
                    input_variables=["context", "question"]
                )

                parallel_chain = RunnableParallel({
                    "context": compression_retriever | RunnableLambda(context),
                    "question":RunnablePassthrough()})

                st.chat_message("user").markdown(query)
                st.session_state.messages.append({"role":"user","content":query})

                final_chain = parallel_chain | prompt | model | parser
                result = final_chain.invoke(query)
            
                st.chat_message("AI").markdown(result)
                st.session_state.messages.append({"role":"AI","content":result})
            else:
                st.warning("No vector Store detected")

