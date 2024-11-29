import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
logging.langsmith("[Project] PDF RAG")


# 캐시 디렉토리 설정
if not os.path.exists(".cache"):
    os.makedirs(".cache")

# 파일 업로드 전용 폴더 설정
if not os.path.exists(".cache/files"):
    os.makedirs(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.makedirs(".cache/embeddings")

st.title("🔥PDF RAG🔥")

# 처음 한번만 실행
if "messages" not in st.session_state:
    # 대화 기록 저장
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드하지 않았다면 체인 생성 안함
    st.session_state["chain"] = None

# 사이드 바
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("Clear Chat", key="clear")

    # 파일 업로드
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    # 모델 선택 메뉴
    selected_model = st.selectbox("Select a model", ["gpt-4o-mini"])

# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state.messages:
        st.chat_message(chat_message["role"]).markdown(chat_message["content"])

# 새로운 메시지 추가
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# 파일을 캐시에 저장(시간이 오래걸리는 작업 처리)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 파일 이름에서 확장자 제거
    file_name = os.path.splitext(file.name)[0]
    embeddings_path = os.path.join(".cache/embeddings", f"{file_name}.faiss")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        api_key=os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"],
    )
    
    # 이미 임베딩된 파일이 있는지 확인
    if os.path.exists(embeddings_path):
        vectorstore = FAISS.load_local(embeddings_path, embeddings)
        return vectorstore.as_retriever()

    # 파일이 없으면 새로 처리
    file_content = file.read()
    file_path = os.path.join(".cache/files", file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 1단계: 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 2단계: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(docs)

    # 3단계, 4단계: 임베딩, 벡터 저장소 생성
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 임베딩 결과 저장
    vectorstore.save_local(embeddings_path)

    # 5단계: 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever()
    
    return retriever

# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 6단계: prompt 생성
    prompt = load_prompt("99.Project/Streamlit/prompts/pdf-rag.yaml", encoding="utf-8")

    # 7단계: LLM 생성
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=model_name,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )

    # 8단계: 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser()
    )
    return chain
    
# 파일이 업로드 되었을 때
if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자 입력
        st.chat_message("user").write(user_input)

        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.warning("파일을 먼저 업로드 해주세요.")