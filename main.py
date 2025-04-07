import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DEFAULT_GPT_MODEL = "gpt-3.5-turbo"
PROMPT_TEMPLATE="""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just return short one-line note that you don't know, don't try to make up an answer and avoid generating lots of suggestions.
            Use the information from the context even if it's only partially related to the question.
            Always cite document names and page numbers when using information. 
            
            {context}
            
            Question: {question}
            Answer:"""


class GameBalanceRAG:
    def __init__(self, persist_directory: str = 'docs/chroma/', 
                 openai_api_key: Optional[str] = None,
                 chunk_size: int = 1000,  # Increased chunk size for better context
                 chunk_overlap: int = 200,  # Increased overlap
                 model_name: str = DEFAULT_GPT_MODEL,
                 temperature: float = 0,
                 game_version: str = "1.0.0",
                 debug_mode: bool = False):  # Added debug mode
        """
        Initialize the Game Balance RAG tool.
        
        Args:
            persist_directory: Directory to persist vector database
            openai_api_key: OpenAI API key (if None, will load from environment)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks to maintain context
            model_name: LLM model name (use chat models like gpt-3.5-turbo, gpt-4o, etc.)
            temperature: LLM temperature parameter
            game_version: Current version of game rules/balance being processed
            debug_mode: If True, print additional debugging information
        """
        # Load environment variables if API key not provided
        if not openai_api_key:
            _ = load_dotenv(find_dotenv())
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.temperature = temperature
        self.game_version = game_version
        self.debug_mode = debug_mode
        
        # Initialize embedding model
        self.embedding = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        # Use CharacterTextSplitter instead for more reliable splitting
        self.splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        
        # Initialize LLM - using ChatOpenAI for chat models
        self.llm = ChatOpenAI(
            model=self.model_name, 
            temperature=self.temperature,
            openai_api_key=self.openai_api_key
        )
        
        # Vector database will be initialized when needed
        self.vectordb = None
        self.retriever = None
        
        # Define metadata fields for document processing
        self.metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The source document filename",
                type="string",
            ),
            AttributeInfo(
                name="page",
                description="The page number in the document",
                type="integer",
            ),
            AttributeInfo(
                name="game_version",
                description="Version of the game rules",
                type="string",
            ),
            AttributeInfo(
                name="category",
                description="Category of game mechanics (combat, economy, skills, etc.)",
                type="string",
            ),
            AttributeInfo(
                name="date_added",
                description="Date when the document was added to the database",
                type="string",
            )
        ]

    def load_documents(self, file_paths: List[str], category: str = "general") -> List:
        """
        Load documents from the provided paths. Supports PDF, TXT, and CSV.
        
        Args:
            file_paths: List of paths to documents
            category: Category to assign to documents for better filtering (e.g., "combat", "economy", "skills")
            
        Returns:
            List of loaded document objects
        """
        docs = []
        for path in file_paths:
            try:
                # Select appropriate loader based on file extension
                if path.lower().endswith('.pdf'):
                    # Use PDFMiner directly for better text extraction
                    loader = PyPDFLoader(path)
                elif path.lower().endswith('.txt'):
                    loader = TextLoader(path)
                elif path.lower().endswith('.csv'):
                    loader = CSVLoader(path)
                else:
                    print(f"Unsupported file type for {path}, skipping...")
                    continue
                
                # Load documents and add metadata
                loaded_docs = loader.load()
                
                # Debug: print the content to verify extraction
                if self.debug_mode:
                    print(f"\nDEBUG - First 500 chars from {path}:")
                    if loaded_docs:
                        print(loaded_docs[0].page_content[:500])
                    else:
                        print("No content extracted!")
                
                for doc in loaded_docs:
                    # Enhance metadata
                    doc.metadata['game_version'] = self.game_version
                    doc.metadata['category'] = category
                    doc.metadata['date_added'] = datetime.now().strftime("%Y-%m-%d")
                    # Source is usually already set by the loader, but ensure it's normalized
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = os.path.basename(path)
                
                docs.extend(loaded_docs)
                print(f"Loaded document: {path} with {len(loaded_docs)} pages/sections")
            except Exception as e:
                print(f"Error loading document {path}: {e}")
        
        print(f"Total documents loaded: {len(docs)}")
        return docs
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into smaller chunks using the configured splitter.
        
        Args:
            documents: List of document objects
            
        Returns:
            List of document chunks
        """
        splits = self.splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(splits)} chunks")
        
        # Debug: print a few chunks to verify proper splitting
        if self.debug_mode:
            print("\nDEBUG - Sample chunks:")
            for i, chunk in enumerate(splits[:2]):  # First 2 chunks
                print(f"\nChunk {i+1} (Length: {len(chunk.page_content)}):")
                print(f"Source: {chunk.metadata.get('source')}, Page: {chunk.metadata.get('page')}")
        
        return splits
    
    def create_vector_database(self, document_chunks: List, force_recreate: bool = False) -> None:
        """
        Create or load the vector database.
        
        Args:
            document_chunks: List of document chunks to embed
            force_recreate: If True, recreate the database even if it exists
        """
        # Check if directory exists and has content
        if os.path.exists(self.persist_directory) and not force_recreate and os.listdir(self.persist_directory):
            print(f"Loading existing vector database from {self.persist_directory}")
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
        else:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            print(f"Creating new vector database with {len(document_chunks)} chunks")
            self.vectordb = Chroma.from_documents(
                documents=document_chunks,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )
            # The persist() method is no longer needed/available in newer Chroma versions
            # The data is automatically persisted when using persist_directory
            
        
        # Update how we access the collection count based on the new Chroma API
        try:
            # For newer versions of Chroma
            collection_count = len(self.vectordb.get())
            print(f"Vector database contains {collection_count} chunks")
        except:
            # Fallback for older versions or different API
            try:
                collection_count = self.vectordb._collection.count()
                print(f"Vector database contains {collection_count} chunks")
            except:
                print("Vector database created successfully (couldn't determine chunk count)")
    
    def setup_retriever(self, retriever_type: str = "basic", 
                      document_content_description: str = "Game balance documentation") -> None:
        """
        Set up the retriever for querying the vector database.
        
        Args:
            retriever_type: Type of retriever to use ("contextual", "self_query", or "basic")
            document_content_description: Description of the document content (for self-query retriever)
        """
        if not self.vectordb:
            raise ValueError("Vector database not initialized. Please create it first.")
        
        if retriever_type == "contextual":
            # Contextual compression retriever extracts relevant parts of documents
            compressor = LLMChainExtractor.from_llm(self.llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vectordb.as_retriever(search_kwargs={"k": 6})  # Increased from 4 to 6
            )
        elif retriever_type == "self_query":
            # Self-query retriever can interpret natural language filters
            self.retriever = SelfQueryRetriever.from_llm(
                self.llm,
                self.vectordb,
                document_content_description,
                metadata_field_info=self.metadata_field_info,
                verbose=True
            )
        else:
            # Basic retriever - using higher k value to get more docs
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        
        print(f"Retriever type '{retriever_type}' has been set up")
    
    def similarity_search(self, query: str, k: int = 5,  # Increased k from 3 to 5
                        metadata_filter: Optional[Dict[str, Any]] = None) -> List:
        """
        Perform a similarity search on the vector database with optional metadata filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional dictionary for filtering results by metadata
            
        Returns:
            List of relevant document chunks
        """
        if not self.vectordb:
            raise ValueError("Vector database not initialized. Please create it first.")
        
        # Debug mode - show what we're searching for
        if self.debug_mode:
            print(f"\nDEBUG - Searching for: '{query}'")
            if metadata_filter:
                print(f"With metadata filter: {metadata_filter}")
        
        if metadata_filter:
            # Search with explicit metadata filter
            docs = self.vectordb.similarity_search_with_score(
                query, 
                k=k,
                filter=metadata_filter
            )
            # Extract just the documents (without scores)
            docs = [doc for doc, _ in docs]
        else:
            # Regular similarity search - use higher k value than default for better coverage
            docs = self.vectordb.similarity_search(query, k=k)
        
        # Debug mode - report results
        if self.debug_mode and docs:
            print(f"\nDEBUG - Found {len(docs)} documents")
            print(f"First document similarity text (sample): {docs[0].page_content[:100]}...")
        elif self.debug_mode:
            print("\nDEBUG - No documents found!")
            
            # Try a direct keyword search to diagnose
            print("Attempting basic keyword analysis in vector database...")
            all_docs = self.vectordb.get()
            if all_docs:
                for doc_content in all_docs['documents'][:5]:  # Check first 5 docs
                    if query.lower() in doc_content.lower():
                        print(f"Found '{query}' in document: {doc_content[:100]}...")
        
        return docs
    
    def keyword_search(self, keyword: str, k: int = 5) -> List:
        """
        Perform a direct keyword search on all documents in the database.
        This is a fallback method when vector search doesn't find matches.
        
        Args:
            keyword: Keyword to search for
            k: Maximum number of results to return
            
        Returns:
            List of documents containing the keyword
        """
        if not self.vectordb:
            raise ValueError("Vector database not initialized. Please create it first.")
            
        # Get all documents from the database
        results = []
        try:
            all_docs = self.vectordb.get()
            
            # Check if we actually got documents back
            if not all_docs or 'documents' not in all_docs or not all_docs['documents']:
                print("No documents found in database for keyword search!")
                return []
                
            # Search for the keyword in each document
            for i, doc_content in enumerate(all_docs['documents']):
                if keyword.lower() in doc_content.lower():
                    # Create a document object with the matching content
                    metadata = {}
                    if 'metadatas' in all_docs and i < len(all_docs['metadatas']):
                        metadata = all_docs['metadatas'][i]
                        
                    from langchain_core.documents import Document
                    doc = Document(page_content=doc_content, metadata=metadata)
                    results.append(doc)
                    
                    if len(results) >= k:
                        break
        except Exception as e:
            print(f"Error in keyword search: {e}")
            
        return results
        
    def hybrid_search(self, query: str, k: int = 5) -> List:
        """
        Perform a hybrid search combining vector similarity, MMR, and keywords.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        if not self.vectordb:
            raise ValueError("Vector database not initialized. Please create it first.")
        
        # Get vector search results
        vector_docs = self.vectordb.similarity_search(query, k=k)
        
        # Get MMR search results (Maximum Marginal Relevance - diverse results)
        mmr_docs = self.vectordb.max_marginal_relevance_search(query, k=k)
        
        # Get keyword search results for any term in the query
        keyword_docs = []
        query_terms = query.lower().split()
        for term in query_terms:
            if len(term) > 3:  # Only search for terms longer than 3 characters
                keyword_docs.extend(self.keyword_search(term, k=2))
        
        # Combine and deduplicate by content
        all_docs = vector_docs + mmr_docs + keyword_docs
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        
        # Convert back to list and limit to k
        result_docs = list(unique_docs.values())
        return result_docs[:k]
    
    def query_llm(self, question: str, k: int = 5, 
                use_retriever: bool = False,
                search_type: str = "hybrid",  # Added search type option
                metadata_filter: Optional[Dict[str, Any]] = None) -> str:
        """
        Query the LLM using data from the vector database.
        
        Args:
            question: Question to ask
            k: Number of relevant documents to retrieve
            use_retriever: Whether to use the configured retriever instead of direct search
            search_type: Type of search to use ("hybrid", "vector", or "keyword")
            metadata_filter: Optional metadata filter for document retrieval
            
        Returns:
            LLM response
        """
        # Get relevant documents
        if use_retriever and self.retriever:
            relevant_docs = self.retriever.get_relevant_documents(question)
        elif search_type == "hybrid":
            relevant_docs = self.hybrid_search(question, k=k)
        elif search_type == "keyword":
            # Split the question into words and search for each one
            keywords = [word for word in question.split() if len(word) > 3]
            if not keywords:
                keywords = [question]  # Use the whole question if no good keywords
            
            relevant_docs = []
            for keyword in keywords:
                relevant_docs.extend(self.keyword_search(keyword, k=2))
                
            # Deduplicate
            unique_docs = {}
            for doc in relevant_docs:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
            relevant_docs = list(unique_docs.values())[:k]
        else:
            relevant_docs = self.similarity_search(question, k=k, metadata_filter=metadata_filter)
        
        # If no relevant docs found with vector search, try keyword search as backup
        if not relevant_docs:
            print("No documents found with primary search method, trying keyword search...")
            # Split the question into words and search for each one
            keywords = [word for word in question.split() if len(word) > 3]
            if not keywords:
                keywords = [question]  # Use the whole question if no good keywords
            
            for keyword in keywords:
                keyword_docs = self.keyword_search(keyword, k=2)
                if keyword_docs:
                    relevant_docs.extend(keyword_docs)
            
            # Deduplicate
            if relevant_docs:
                unique_docs = {}
                for doc in relevant_docs:
                    if doc.page_content not in unique_docs:
                        unique_docs[doc.page_content] = doc
                relevant_docs = list(unique_docs.values())[:k]
        
        # If still no relevant docs, return a message
        if not relevant_docs:
            return "I couldn't find any relevant information in the documents to answer your question about '" + question + "'. Please try reformulating your question or check if the documents contain this information."
        
        # Format context for the LLM
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            # Include metadata for context
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', 'Unknown page')
            category = doc.metadata.get('category', 'Unknown category')
            
            context_part = f"Document {i+1} [{source}, Page {page}, Category: {category}]:\n{doc.page_content}"
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt with context
        prompt = f"""
        You are a game balance expert assistant. Use the following information to answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - If the provided information contains the answer or related information, provide a thorough response.
        - Even partially related information should be used to form the best possible answer.
        - Cite the specific document and page numbers when referring to information.
        - Format your response in a clear, structured way.
        - Focus on game balance implications in your answer.
        - If the information is truly not relevant, explain why it doesn't address the question.
        
        Answer:
        """
        
        # Query the LLM
        response = self.llm.invoke(prompt)
        
        # For ChatOpenAI models, we need to extract the content from the response
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    # def query_with_chain(self, question: str) -> Dict[str, Any]:
    #     """
    #     Use a RetrievalQA chain for more structured responses.
    #
    #     Args:
    #         question: Question to ask
    #
    #     Returns:
    #         Dictionary with response and source documents
    #     """
    #     if not self.vectordb:
    #         raise ValueError("Vector database not initialized.")
    #
    #     # Define a custom prompt for the chain
    #     custom_prompt = PromptTemplate(
    #         PROMPT_TEMPLATE,
    #         input_variables=["context", "question"]
    #     )
    #
    #     # Create a QA chain that returns source documents
    #     qa_chain = RetrievalQA.from_chain_type(
    #         llm=self.llm,
    #         chain_type="stuff",  # Combines all documents into single context
    #         retriever=self.vectordb.as_retriever(search_kwargs={"k": 5}),
    #         chain_type_kwargs={"prompt": custom_prompt},
    #         return_source_documents=True,
    #         verbose=True
    #     )
    #
    #     # Run the chain
    #     result = qa_chain({"query": question})
    #     return result
    
    def query_structured(self, question: str, output_schema: Dict) -> Dict:
        pass
    #     """
    #     Get structured JSON responses from the LLM.
    #
    #     Args:
    #         question: Question to ask
    #         output_schema: Schema defining the expected output format
    #
    #     Returns:
    #         Structured response as a dictionary
    #     """
    #     # Get relevant documents with hybrid search for better recall
    #     relevant_docs = self.hybrid_search(question, k=7)
    #
    #     # If no docs found, try direct keyword search
    #     if not relevant_docs:
    #         # Split the question into words and search for each one
    #         keywords = [word for word in question.split() if len(word) > 3]
    #         if not keywords:
    #             keywords = [question]  # Use the whole question if no good keywords
    #
    #         for keyword in keywords:
    #             keyword_docs = self.keyword_search(keyword, k=2)
    #             if keyword_docs:
    #                 relevant_docs.extend(keyword_docs)
    #
    #     # If still no docs found, return error
    #     if not relevant_docs:
    #         return {
    #             "error": "No relevant information found",
    #             "mechanic_name": question,
    #             "description": "No information found in the provided documents.",
    #             "balance_implications": "Unable to determine without document information.",
    #             "recommended_values": "No data available."
    #         }
    #
    #     context = "\n\n".join([doc.page_content for doc in relevant_docs])
    #
    #     # Create prompt for structured output
    #     prompt = f"""
    #     You are a game balance expert assistant. Use the following information to answer the question.
    #
    #     Context:
    #     {context}
    #
    #     Question: {question}
    #
    #     Instructions:
    #     - Use the provided information to the best of your ability.
    #     - If information is incomplete, note this but still provide what you can based on context.
    #     - Make educated inferences about game balance if direct information is lacking.
    #
    #     Return your answer as a JSON object following this schema:
    #     {output_schema}
    #
    #     Only return the JSON object, no other text.
    #     """
    #
    #     # Parse the response as JSON
    #     response = self.llm.invoke(prompt)
    #
    #     # For ChatOpenAI, we need to extract the content
    #     if hasattr(response, 'content'):
    #         response_text = response.content
    #     else:
    #         response_text = str(response)
    #
    #     # Find JSON in response (handles cases where LLM adds extra text)
    #     json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    #     if json_match:
    #         json_str = json_match.group(1)
    #     else:
    #         json_str = response_text
    #
    #     try:
    #         return json.loads(json_str)
    #     except json.JSONDecodeError:
    #         return {
    #             "error": "Could not parse response as JSON",
    #             "raw_response": str(response),
    #             "mechanic_name": question,
    #             "description": "Error processing the response.",
    #             "balance_implications": "Unable to format structured data.",
    #             "recommended_values": "Please try again with a different query."
    #         }
    
    def process_documents_and_create_db(self, file_paths: List[str], 
                                      category: str = "general",
                                      force_recreate: bool = False,
                                      retriever_type: str = "basic") -> None:
        """
        Load documents, split them, and create the vector database in one go.
        
        Args:
            file_paths: List of paths to documents
            category: Category to assign to these documents
            force_recreate: If True, recreate the database even if it exists
            retriever_type: Type of retriever to set up
        """
        print(f"Starting document processing pipeline for {len(file_paths)} documents")
        
        # Load documents
        print("Loading documents...")
        docs = self.load_documents(file_paths, category=category)
        
        # Split documents
        print("Splitting documents into chunks...")
        splits = self.split_documents(docs)
        
        # Create vector database
        print("Creating/updating vector database...")
        self.create_vector_database(splits, force_recreate=force_recreate)
        
        # Setup retriever
        print(f"Setting up {retriever_type} retriever...")
        self.setup_retriever(retriever_type=retriever_type)
        
        print("Document processing complete!")
    
    def update_vector_database(self, file_paths: List[str], category: str = "general") -> None:
        """
        Add new documents to an existing vector database without recreating it.
        
        Args:
            file_paths: List of paths to new documents
            category: Category to assign to these documents
        """
        if not self.vectordb:
            raise ValueError("Database doesn't exist yet. Use process_documents_and_create_db instead.")
        
        print(f"Updating vector database with {len(file_paths)} new documents")
        
        # Load and process new documents
        new_docs = self.load_documents(file_paths, category=category)
        new_chunks = self.split_documents(new_docs)
        
        # Add to existing database
        print(f"Adding {len(new_chunks)} new chunks to database")
        self.vectordb.add_documents(new_chunks)
        # No need to call persist() in newer versions
        
        # Update count information with proper handling for API changes
        try:
            # For newer versions of Chroma
            collection_count = len(self.vectordb.get())
            print(f"Database updated, now contains {collection_count} chunks")
        except:
            # Fallback for older versions or different API
            try:
                collection_count = self.vectordb._collection.count()
                print(f"Database updated, now contains {collection_count} chunks")
            except:
                print("Database updated successfully (couldn't determine chunk count)")


def main():
    """Example usage of the GameBalanceRAG class."""
    import json
    
    # PDF paths
    file_paths = [
        "docs/balance/Базовый баланс.pdf",
        "docs/balance/Калькулятор стоимости улучшений.pdf",
    ]
    
    # Initialize the RAG tool with some parameters
    rag_tool = GameBalanceRAG(
        chunk_size=300,
        chunk_overlap=100,
        model_name=DEFAULT_GPT_MODEL,
        temperature=0.0,           
        game_version="1.0.0",
        debug_mode=False
    )
    
    # Process documents and create database - force recreate to ensure fresh start
    rag_tool.process_documents_and_create_db(
        file_paths, 
        category="combat_mechanics",
        force_recreate=False,
        retriever_type="basic"
    )
    
    # Using metadata filtering example
    combat_filter = {"category": "combat_mechanics"}
    
    # Ask a question
    question = input("Enter your question:")
    if not question:
        question = "За яким принципом рахується кількість ворогів, що буде зачеплено при вибуху снаряда"

    print(f"\nQuestion: {question}")
    
    # Try hybrid search instead of regular similarity search
    docs = rag_tool.hybrid_search(question, k=5)

    if rag_tool.debug_mode:
        print(f"\nFound {len(docs)} relevant documents:")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    # Query the LLM with hybrid search
    response = rag_tool.query_llm(question, search_type="hybrid")
    print(f"\nLLM Response:\n{response}")
    

    need_structured = ""
    while need_structured not in ["Y", "N"]:
        need_structured = input("Do you need structured output? Y|N: ")
    if need_structured == "Y":
        schema = {
            "type": "object",
            "properties": {
                "mechanic_name": {"type": "string"},
                "description": {"type": "string"},
                "balance_implications": {"type": "string"},
                "recommended_values": {"type": "string"}
            }
        }
    
        structured_response = rag_tool.query_structured(question, schema)
        print("\nStructured Response:")
        print(json.dumps(structured_response, indent=2))


if __name__ == "__main__":
    main()