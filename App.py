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
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
app = Flask(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
file_path = "./doccuments/file/uploaded_20240910-03.pdf"

# 推論結果格納dirの作成
   # doccuments: 登録する文書
   # splits: text化したファイルの格納先
if not os.path.exists("./doccuments"):
    os.makedirs("./doccuments/file")
if not os.path.exists("splits"):
    os.makedirs("./splits")
class LLM_Loader:
    @app.route('/')
    def index():
        files = [file for file in os.listdir('./doccuments/file/')]
        return render_template('index.html', files=files)

    @app.route('/upload', methods=["POST"])
    def upload():
        # 現在の仕様:uploadできるファイルは一件のみ 
            # すでにファイルがdoccumentsにuploadされていたら削除
        for f in os.listdir("./doccuments/file/"):
            print(f)
            os.remove(os.path.join("./doccuments/file/", f))
        for f in os.listdir("./doccuments/file/"):
            os.remove(os.path.join("./doccuments/splits/", f))
        file = request.files.get('file')
        file_name = file.filename
        file_path = os.path.join('./doccuments/file', file_name)
        file.save(file_path)
        file.close()
        return redirect(url_for('index'))

    @app.route('/download/<string:file>')
    def download(file):
        return send_from_directory('doccuments/file', file, as_attachment=True)
    
    @app.route('/execute', methods = ['POST'])
    #LLM推論実行エンドポイント
     # file配下の文書をテキストに変換
    def execute():
        query = request.form.get('query')
        file_name = [file for file in os.listdir('./doccuments/file')][0]
        file_path = os.path.join('./doccuments/file', file_name)
        # <doccumentInteligence>uploadされている文書をtextファイルとして変換
        loader = AzureAIDocumentIntelligenceLoader(file_path=file_path, api_key = doc_intelligence_key, api_endpoint = doc_intelligence_endpoint, api_model="prebuilt-layout")
        docs = loader.load()
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs_string = docs[0].page_content
        splits = text_splitter.split_text(docs_string)
        for i, split in enumerate(splits):
            with open(f"doccuments/splits/split_{i}.txt", "w",encoding='UTF-8') as f:
                f.write(split.page_content)

        # load embedding model
         # AzureSearchのデータベースにsplit配下のテキストデータを格納
        aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2024-02-15-preview",  # e.g., "2023-12-01-preview"
        )
        vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
        vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        index_name: str = "sios_sample_index"
        vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=vector_store_password,
            index_name=index_name,
            embedding_function=aoai_embeddings.embed_query,
        )
        loader = DirectoryLoader("doccuments/splits")
        documents = loader.load()

        vector_store.add_documents(documents)


        # load embedding model
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
        # OpenAI,AzureSearchにquery(質問)を入力し、db格納のデータから類似度が高いものを返答する。
        retriever = vector_store.as_retriever()
        answer_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        res = answer_chain.invoke(query)
        return render_template('index.html', result = res)




    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
        