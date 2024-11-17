from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv
from langchain_core.runnables import chain as chain_decorator


class LLM_Loader:
    def __init__(self):
        load_dotenv(verbose=True)
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path)
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        self.file_path = "./doccuments/file/uploaded_20240910-03.pdf"

        pass

    def __call__(self):
        loaded_file = self.file_loader(self.file_path)
        self.chanked_text_save(loaded_file)
        self.embedding_text()
        self.create_template()
        pass
    

    def file_loader(self,file_path):
        loader = AzureAIDocumentIntelligenceLoader(file_path=file_path, api_key = self.doc_intelligence_key, api_endpoint = self.doc_intelligence_endpoint, api_model="prebuilt-layout")
        docs = loader.load()
        return docs

    def chanked_text_save(self, loaded_file):
        # Split the document into chunks base on markdown headers.
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs_string = loaded_file[0].page_content
        splits = text_splitter.split_text(docs_string)
        print(splits)
        for i, split in enumerate(splits):
            with open(f"doccuments/splits/split_{i}.txt", "w",encoding='UTF-8') as f:
                f.write(split.page_content)


    def embedding_text(self):
        aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2024-02-15-preview",  # e.g., "2023-12-01-preview"
        )
        vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
        vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        index_name: str = "pdf_index"
        vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=vector_store_password,
            index_name=index_name,
            embedding_function=aoai_embeddings.embed_query,
        )
        loader = DirectoryLoader("doccuments/splits")
        documents = loader.load()
        try:
            yield vector_store.add_documents(documents)
        finally:
            vector_store.close()
    
    def create_template(self):
        aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2024-02-15-preview",  # e.g., "2023-12-01-preview"
        )
        vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
        vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        index_name: str = "pdf_index"
        vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=vector_store_password,
            index_name=index_name,
            embedding_function=aoai_embeddings.embed_query,
        )
        llm = AzureChatOpenAI(
        openai_api_version="2024-02-01",  # e.g., "2023-12-01-preview"
        azure_deployment="gpt-4o",
        temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(
            """質問に対して、関連情報を参照に回答してください。
            関連する情報を参照しても分からない場合は、「分かりません」と回答してください。"""
        ),
        HumanMessagePromptTemplate.from_template(
            """ 関連情報：{context}
            
            ## 質問：{question}
            ## 回答： """
        )
        ]
        )
    
        # Documetを連結する
        @chain_decorator
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # answerを得るためのchain
        retriever = vector_store.as_retriever()
        answer_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(answer_chain.invoke("本資料は誰についてかかれていますか？"))


    


    
if __name__ == "__main__":
    LLM = LLM_Loader()
    LLM()
        