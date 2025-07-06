from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Khai bao bien 
pdf_data_path = "data"
vector_dp_path = "vectorstores/db_faiss"

# Ham 1. Tao ra vector DB tu 1 doan text
def create_db_from_text():
    raw_text = "Trường Đại học Khoa học – Đại học Huế là một trong những cơ sở đào tạo và nghiên cứu hàng đầu tại khu vực miền Trung và Tây Nguyên. Được thành lập từ năm 1957, trường có bề dày truyền thống trong giảng dạy các ngành khoa học tự nhiên, xã hội và nhân văn. Với đội ngũ giảng viên giàu kinh nghiệm, cơ sở vật chất hiện đại và môi trường học tập năng động, Trường Đại học Khoa học luôn là lựa chọn uy tín của sinh viên trong và ngoài nước. Trường hiện tọa lạc tại số 77 Nguyễn Huệ, thành phố Huế – trung tâm văn hóa, giáo dục lớn của cả nước."

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len 
    )

    chunks = text_splitter.split_text(raw_text)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file= "model/tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
    
    # Dua vao Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_dp_path)
    return db

def create_dp_from_files():
    # Khai bao loader de quet toan bo thu muc data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size = 512, chunk_overlap = 50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = GPT4AllEmbeddings(model_file = "model/tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
    dp = FAISS.from_documents(chunks, embedding_model)
    dp.save_local(vector_dp_path)
    return dp

# create_db_from_text()
create_dp_from_files()