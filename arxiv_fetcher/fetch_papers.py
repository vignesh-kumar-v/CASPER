import urllib.request
import xml.etree.ElementTree as ET
import os
import json
from pathlib import Path

# Note: You will need to install pymupdf for text extraction: pip install pymupdf
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file using PyMuPDF.
    Handles potential layout and empty page issues by attempting to grab all text blocks.
    """
    if fitz is None:
        print("Error: PyMuPDF (fitz) not installed. Please run 'pip install pymupdf'")
        return ""

    text_content = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # get_text() tries to reconstruct the layout reasonably well
            page_text = page.get_text("text")
            if page_text.strip():
                text_content.append(page_text.strip())
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    
    return "\n".join(text_content)

def chunk_text(text, window_tokens=500, overlap_tokens=50):
    """
    Performs sliding window fragmentation. 
    Uses a word-to-token ratio to convert token counts into approximate word counts.
    """
    # Industry standard: ~1 token is ~0.75 words (or more accurately, 1 word is approx 1.33 tokens)
    # If we want a window of 500 tokens, and 1 token = 0.75 words, 
    # then window_size_words = 500 * 0.75 = 375 words.
    TOKEN_TO_WORD_RATIO = 0.75
    
    window_size_words = int(window_tokens * TOKEN_TO_WORD_RATIO)
    overlap_words = int(overlap_tokens * TOKEN_TO_WORD_RATIO)

    if window_size_words <= overlap_words:
        # Safety check to prevent infinite loops or invalid windows
        window_size_words = max(1, window_size_words)
        overlap_words = 0

    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        # Extract chunk
        chunk_end = min(i + window_size_words, len(words))
        chunk_content = " ".join(words[i : chunk_end])
        chunks.append(chunk_content)
        
        # Move pointer forward by (window - overlap)
        step = window_size_words - overlap_words
        if step <= 0:
            break
        i += step
        
    return chunks

def fetch_papers(query, k=4, output_text_file="all_papers_text.txt", output_chunks_file="all_papers_chunks.json"):
    """
    Fetches papers from Arxiv based on a search query, sorts by relevance, 
    extracts text with detailed metadata packaging (title and index), 
    and saves results into a single text file and a structured JSON.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = f"?search_query=all:{query}&start=0&max_results={k}&sortBy=relevance&sortOrder=descending"
    url = base_url + params

    print(f"Searching for: {query}")
    
    try:
        with urllib.request.urlopen(url) as response:
            xml_data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching from Arxiv API: {e}")
        return

    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    download_dir = Path("arxiv_fetcher/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    # We will store tuples of (title, file_path)
    papers_to_process = []
    papers_found = 0

    for entry in root.findall('atom:entry', ns):
        papers_found += 1
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        paper_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
        
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.get('type') == 'application/pdf':
                pdf_url = link.get('href')
                break
        
        if not pdf_url:
             for link in entry.findall('atom:link', ns):
                if link.get('href').endswith('.pdf'):
                    pdf_url = link.get('href')
                    break

        if not pdf_url:
            abstract_url = entry.find('atom:id', ns).text
            if '/abs/' in abstract_url:
                pdf_url = abstract_url.replace('/abs/', '/pdf/') + '.pdf'

        if pdf_url:
            filename = f"{paper_id}.pdf"
            file_path = download_dir / filename

            if file_path.exists():
                print(f"Cache hit: {filename} ({title[:50]}...)")
            else:
                print(f"Downloading: {title[:50]}...")
                try:
                    with urllib.request.urlopen(pdf_url) as pdf_response:
                        with open(file_path, 'wb') as f:
                            f.write(pdf_response.read())
                    print(f"Saved: {filename}")
                except Exception as e:
                    print(f"Failed to download {pdf_url}: {e}")
            
            if file_path.exists():
                papers_to_process.append({"title": title, "path": file_path})

        else:
            print(f"Could not find PDF URL for paper: {title[:50]}...")

    if papers_found == 0:
        print("No papers found.")
        return

    print(f"\nFinished downloading. Processing up to {k} papers...\n")

    if not papers_to_process:
        print("No PDF files were available for processing.")
        return

    all_chunks_data = []
    full_text_accumulator = []
    global_chunk_index = 0

    for paper in papers_to_process:
        title = paper["title"]
        path = paper["path"]
        
        print(f"Extracting and chunking: {title[:50]}...")
        text = extract_text_from_pdf(path)
        
        if text:
            full_text_accumulator.append(text)
            # Generate chunks for this specific paper
            paper_chunks = chunk_text(text, window_tokens=500, overlap_tokens=50)
            
            for chunk_content in paper_chunks:
                chunk_entry = {
                    "text": chunk_content,
                    "metadata": {
                        "title": title,
                        "index": global_chunk_index
                    }
                }
                all_chunks_data.append(chunk_entry)
                global_chunk_index += 1

    # Save the full text file
    if full_text_accumulator:
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text_accumulator))
        print(f"Successfully wrote full text content to {output_text_file}")

        # Save the structured chunks JSON
        with open(output_chunks_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(all_chunks_data)} structured chunks to {output_chunks_file}")
    else:
        print("No text could be extracted from the PDFs.")

    print(f"\nFinished processing. Processed {len(papers_to_process)} papers.")

if __name__ == "__main__":
    import sys
    # Sample query provided for testing: 'attention mechanism'
    query_input = sys.argv[1] if len(sys.argv) > 1 else "attention mechanism"
    fetch_papers(query_input)
