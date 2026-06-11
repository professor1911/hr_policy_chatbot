# Shared helper functions for the HR AI Assistant
import re
import pandas as pd
import os
import csv
import logging
from datetime import datetime
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from src.config.config import GROQ_API_KEY, MODEL_CONFIG

# Set up logging for this module
logger = logging.getLogger(__name__)

def get_user_intent(prompt: str) -> str:
    """Classify user input into intent categories."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    chat_model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_CONFIG["classifier_model"],
        temperature=0.1,
    )

    # Read classifier prompt from file - Adjusted path for backend
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'prompts', 'classifier_prompt.txt')

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            classifier_prompt_text = f.read()
    except FileNotFoundError:
        logger.error(f"Classifier prompt file not found at {prompt_path}")
        raise FileNotFoundError(f"Classifier prompt file not found at {prompt_path}")

    # Create classification prompt using the prompt from file
    classification_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=classifier_prompt_text),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])

    chain = classification_prompt | chat_model

    response = chain.invoke({"human_input": prompt})

    raw_response = response.content.strip()
    if "</think>" in raw_response:
        raw_response = raw_response.split("</think>")[-1].strip()

    # logger.info(f"Classifier raw response: '{raw_response}'")  # Commented out for cleaner output

    # Extract category from response
    category = raw_response.upper()
    if "HR_POLICY" in category:
        final_intent = "HR_POLICY"
    elif "MALICIOUS_INPUT" in category:
        final_intent = "MALICIOUS_INPUT"
    else:
        final_intent = "CASUAL_CONVERSATION"

    # logger.info(f"Final classified intent: {final_intent}")  # Commented out for cleaner output
    return final_intent

def safe_convert(value, default=0, convert_to=int):
    """
    Safely convert a value to the specified type (int or float).
    Handles various formats and returns a default if conversion fails.
    """
    if value is None:
        return default
    if isinstance(value, convert_to):
        return value
    if isinstance(value, (int, float)):
        return convert_to(value)
    if isinstance(value, str):
        if convert_to == float:
            match = re.search(r'(\d+\.\d+|\d+)', value)
            if match:
                try:
                    return convert_to(match.group(1))
                except (ValueError, TypeError):
                    pass
        else:
            match = re.search(r'(\d+)', value)
            if match:
                try:
                    return convert_to(match.group(1))
                except (ValueError, TypeError):
                    pass
    logging.debug(f"Conversion failed for '{value}', using default: {default}")
    return default

def load_feedback(feedback_file='feedback_log.csv'):
    """Load feedback from CSV and return as a list of dicts."""
    feedback_path = os.path.join(os.path.dirname(__file__), '..', '..', feedback_file) # Adjusted path
    feedback = []
    if os.path.exists(feedback_path):
        with open(feedback_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 4:
                    feedback.append({
                        'timestamp': row[0],
                        'question': row[1],
                        'answer': row[2],
                        'feedback': row[3]
                    })
    return feedback

def get_feedback_penalty(query, doc, feedback):
    """Return a penalty or boost for a doc based on feedback for similar questions."""
    penalty = 0
    for entry in feedback:
        if doc.metadata.get('source', '') in entry['answer']:
            if entry['feedback'] == 'not_helpful' and any(word in entry['question'].lower() for word in query.lower().split()):
                penalty -= 2
            elif entry['feedback'] == 'helpful' and any(word in entry['question'].lower() for word in query.lower().split()):
                penalty += 2
    return penalty

def suggest_next_steps(query):
    return ("If you need more information, try rephrasing your question, consult the HR team, or refer to the full document for details.")

def load_documents(files) -> List[Document]:
    """Load documents from uploaded files."""
    documents = []
    for file in files:
        try:
            # Save the uploaded file temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())

            # Load based on file type
            if file.name.endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
            elif file.name.endswith('.txt'):
                loader = TextLoader(temp_path)
            else:
                loader = UnstructuredFileLoader(temp_path)

            documents.extend(loader.load())

            # Clean up temporary file
            os.remove(temp_path)
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    return documents

def log_feedback(question: str, answer: str, feedback: str, feedback_file='feedback_log.csv'):
    """Log user feedback to a CSV file."""
    feedback_path = os.path.join(os.path.dirname(__file__), '..', '..', feedback_file) # Adjusted path

    # Create the file with headers if it doesn't exist
    file_exists = os.path.exists(feedback_path)

    with open(feedback_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header if file is new
        if not file_exists:
            writer.writerow(['timestamp', 'question', 'answer', 'feedback'])

        # Write the feedback entry
        writer.writerow([
            datetime.now().isoformat(),
            question,
            answer,
            feedback
        ])

    logger.info(f"Feedback logged: {feedback} for question: {question[:50]}...")