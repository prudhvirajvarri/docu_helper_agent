import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

output_filename = 'docs_text.txt'

def scrape_and_chunk(urls):
    all_text = ""
    print("Scraping URLs")

    for url in urls:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        main_content = soup.find('div', 'bd-content')
        
        if main_content:
            text_content = main_content.get_text(separator='\n', strip=True)
            all_text += text_content + "\\n\\n---Page Break ---\\n\\n"
        else:
            print(f"Main content not found for {url}")

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(all_text)
    print(f"Scraped content saved to {output_filename}")
    """
    with open(output_filename, 'r', encoding='utf-8') as f:
        docs_text = f.read()
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )

    chunks = text_splitter.split_text(all_text)
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks

def main():
    
    num_urls = int(input("Enter the number of URLs to scrape: "))
    urls = []
    for i in range(num_urls):
        url = input(f"Enter URL {i +1}: ")
        urls.append(url)
    
    chunks = scrape_and_chunk(urls)
    
    embedding_model = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_texts(
        texts = chunks,
        embedding = embedding_model,
        persist_directory = "chroma_db_2"
    )

if __name__ == "__main__":
    main()