# rag_chain.py
import logging
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
import time

# Configure logging
logger = logging.getLogger(__name__)

def create_qa_chain(vector_store, groq_client, model_name="llama-3.3-70b-versatile"):
    """Create a simple Q&A function using Groq."""

    # Load HR prompt template from external file for easier editing
    def load_prompt_template():
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "model_prompt.txt")
        logger.info(f"Loading prompt template from: {prompt_path}")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
                logger.info(f"Successfully loaded prompt template ({len(template)} characters)")
                return template
        except FileNotFoundError:
            logger.error(f"Model prompt file not found at {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading prompt template: {e}")
            raise

    hr_template = load_prompt_template()

    def process_query(query_data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Process a query using the Groq client and vector store.
        Returns a tuple of (answer, sources)."""
        start_time = time.time()
        
        try:
            # Extract question from input
            question = query_data.get("question", "")
            chat_history = query_data.get("chat_history", "")
            
            logger.info(f"Processing query: '{question[:100]}{'...' if len(question) > 100 else ''}'")
            logger.debug(f"Chat history length: {len(chat_history)} characters")

            # Get relevant documents
            logger.info("Searching for relevant documents in vector store")
            search_start = time.time()
            relevant_docs = vector_store.similarity_search(question, k=6)
            search_time = time.time() - search_start
            
            logger.info(f"Found {len(relevant_docs)} relevant documents in {search_time:.2f}s")
            
            # Log document details
            for i, doc in enumerate(relevant_docs):
                doc_source = doc.metadata.get('source', 'Unknown')
                doc_length = len(doc.page_content)
                logger.debug(f"Document {i+1}: {os.path.basename(doc_source)} ({doc_length} chars)")
            
            # Extract document content
            doc_texts = [doc.page_content for doc in relevant_docs]
            context = "\n\n---\n\n".join(doc_texts)
            logger.debug(f"Combined context length: {len(context)} characters")

            # Format the prompt
            logger.info("Formatting prompt with context and chat history")
            formatted_prompt = hr_template.format(
                context=context,
                question=question,
                chat_history=chat_history
            )
            logger.debug(f"Final prompt length: {len(formatted_prompt)} characters")

            logger.info(f"Calling Groq API with model: {model_name}")
            api_start = time.time()
            messages = [{"role": "user", "content": formatted_prompt}]

            response = groq_client.chat_completion(
                messages=messages,
                model=model_name,
                temperature=0.2,
                max_tokens=1500,
            )
            api_time = time.time() - api_start
            logger.info(f"Groq API call completed in {api_time:.2f}s")

            # Extract answer from response
            if response.get("choices") and len(response["choices"]) > 0:
                answer = response["choices"][0]["message"]["content"]
                logger.info(f"Generated answer length: {len(answer)} characters")
                logger.debug(f"Answer preview: {answer[:150]}{'...' if len(answer) > 150 else ''}")
            else:
                logger.warning("No answer generated from Groq response")
                answer = "No answer generated. Please try again."

            # Get source documents
            sources = list(dict.fromkeys(
                os.path.basename(doc.metadata.get('source', 'Unknown document'))
                for doc in relevant_docs
            ))
            logger.info(f"Source documents: {', '.join(sources)}")

            # Log total processing time
            total_time = time.time() - start_time
            logger.info(f"Query processing completed in {total_time:.2f}s total")

            return answer, sources

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error in process_query after {total_time:.2f}s: {e}")
            logger.error(f"Query that failed: {query_data.get('question', 'No question provided')}")
            return f"An error occurred: {str(e)}", []

    return process_query