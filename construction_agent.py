#!/usr/bin/env python3
"""
AI Construction Scheduling Agent

This script creates a sophisticated AI agent for construction scheduling that:
- Uses llama-3.3-70b via Cerebras API with automatic key rotation
- Implements RAG pipeline for document analysis
- Follows ReAct framework for reasoning and acting
- Generates detailed project schedules from construction documents

Required Dependencies:
pip install langchain langchain-community chromadb sentence-transformers PyMuPDF pytesseract requests python-dotenv

Required System Dependencies:
- Tesseract OCR: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)
"""

import os
import json
import time
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import requests
from datetime import datetime, timedelta

# LangChain imports
from langchain.llms.base import LLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from langchain.schema import BaseMessage
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document processing imports
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# =============================================================================
# 1. SMART API KEY ROTATOR
# =============================================================================

class CerebrasAPIKeyRotator:
    """
    Manages multiple Cerebras API keys and rotates them when rate limits are hit.
    
    This class ensures resilient API access by automatically switching to the next
    available key when a 429 (Too Many Requests) error is encountered.
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize the key rotator with a list of API keys.
        
        Args:
            api_keys: List of Cerebras API keys
        """
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        
        self.api_keys = api_keys
        self.current_index = 0
        self.total_keys = len(api_keys)
        
        print(f"[KEY ROTATOR] Initialized with {self.total_keys} API keys")
    
    def get_key(self) -> str:
        """Get the currently active API key."""
        return self.api_keys[self.current_index]
    
    def rotate_key(self) -> bool:
        """
        Rotate to the next API key.
        
        Returns:
            True if rotation was successful, False if all keys have been exhausted
        """
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % self.total_keys
        
        print(f"[KEY ROTATOR] Rotated from key {old_index + 1} to key {self.current_index + 1}")
        
        # Return False if we've cycled through all keys
        return self.current_index != old_index or self.total_keys == 1
    
    def reset(self):
        """Reset to the first API key."""
        self.current_index = 0
        print("[KEY ROTATOR] Reset to first API key")

# =============================================================================
# 2. RAG PIPELINE & DOCUMENT PROCESSING
# =============================================================================

class DocumentProcessor:
    """
    Handles loading, processing, and indexing of construction documents.
    
    This class implements the RAG pipeline that converts PDF documents into
    a searchable vector database using embeddings.
    """
    
    def __init__(self):
        """Initialize the document processor with embedding model and text splitter."""
        print("[RAG] Initializing document processor...")
        
        # Initialize embedding model
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize text splitter for intelligent chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vectorstore = None
        print("[RAG] Document processor initialized successfully")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using both direct text extraction and OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        print(f"[RAG] Processing PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                # Try direct text extraction first
                text = page.get_text()
                
                # If no text found, try OCR on the page image
                if not text.strip():
                    print(f"[RAG] No direct text found on page {page_num + 1}, attempting OCR...")
                    
                    # Convert page to image
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    try:
                        text = pytesseract.image_to_string(img)
                    except Exception as ocr_error:
                        print(f"[RAG] OCR failed for page {page_num + 1}: {ocr_error}")
                        text = ""
                
                full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            doc.close()
            print(f"[RAG] Successfully extracted {len(full_text)} characters from {total_pages} pages")
            return full_text
            
        except Exception as e:
            print(f"[RAG] Error processing PDF {pdf_path}: {e}")
            return ""
    
    def process_documents(self, document_paths: List[str]) -> bool:
        """
        Process multiple documents and create a searchable vector store.
        
        Args:
            document_paths: List of paths to PDF documents
            
        Returns:
            True if processing was successful, False otherwise
        """
        if not document_paths:
            print("[RAG] No documents provided for processing")
            return False
        
        print(f"[RAG] Processing {len(document_paths)} documents...")
        
        all_texts = []
        all_metadatas = []
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                print(f"[RAG] Warning: Document not found: {doc_path}")
                continue
            
            # Extract text from the document
            text = self.extract_text_from_pdf(doc_path)
            
            if text.strip():
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_metadatas.append({
                        "source": os.path.basename(doc_path),
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    })
                
                print(f"[RAG] Created {len(chunks)} chunks from {os.path.basename(doc_path)}")
        
        if not all_texts:
            print("[RAG] No text content extracted from any documents")
            return False
        
        # Create vector store
        try:
            self.vectorstore = Chroma.from_texts(
                texts=all_texts,
                embedding=self.embeddings,
                metadatas=all_metadatas,
                persist_directory="./chroma_db"
            )
            
            print(f"[RAG] Created vector store with {len(all_texts)} chunks")
            return True
            
        except Exception as e:
            print(f"[RAG] Error creating vector store: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the document vector store for relevant information.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.vectorstore:
            print("[RAG] Vector store not initialized. Please process documents first.")
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            print(f"[RAG] Found {len(formatted_results)} relevant chunks for query: '{query[:50]}...'")
            return formatted_results
            
        except Exception as e:
            print(f"[RAG] Error searching documents: {e}")
            return []

# =============================================================================
# 3. STATE MANAGER (AGENT'S SCRATCHPAD)
# =============================================================================

@dataclass
class Task:
    """Represents a single task in the construction schedule."""
    id: str
    name: str
    description: str
    duration_days: int
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    resource_requirements: List[str] = field(default_factory=list)
    
class ScheduleState:
    """
    Manages the agent's working memory for building the construction schedule.
    
    This class serves as the agent's scratchpad, storing tasks, dependencies,
    and schedule information that the agent builds incrementally.
    """
    
    def __init__(self):
        """Initialize an empty schedule state."""
        self.tasks: Dict[str, Task] = {}
        self.project_name = ""
        self.project_start_date = None
        self.project_duration = 0
        
        print("[SCHEDULE] Schedule state initialized")
    
    def set_project_info(self, name: str, start_date: str = None):
        """Set basic project information."""
        self.project_name = name
        self.project_start_date = start_date
        print(f"[SCHEDULE] Project info set: {name}")
    
    def add_task(self, task_id: str, name: str, description: str, duration_days: int) -> bool:
        """
        Add a new task to the schedule.
        
        Args:
            task_id: Unique identifier for the task
            name: Task name
            description: Detailed task description
            duration_days: Task duration in days
            
        Returns:
            True if task was added successfully
        """
        if task_id in self.tasks:
            print(f"[SCHEDULE] Task {task_id} already exists, updating...")
        
        self.tasks[task_id] = Task(
            id=task_id,
            name=name,
            description=description,
            duration_days=duration_days
        )
        
        print(f"[SCHEDULE] Added task: {task_id} - {name} ({duration_days} days)")
        return True
    
    def add_dependency(self, successor_id: str, predecessor_id: str) -> bool:
        """
        Add a dependency relationship between two tasks.
        
        Args:
            successor_id: ID of the task that depends on the predecessor
            predecessor_id: ID of the task that must be completed first
            
        Returns:
            True if dependency was added successfully
        """
        if successor_id not in self.tasks:
            print(f"[SCHEDULE] Error: Successor task {successor_id} not found")
            return False
        
        if predecessor_id not in self.tasks:
            print(f"[SCHEDULE] Error: Predecessor task {predecessor_id} not found")
            return False
        
        # Add to predecessor's successors
        if successor_id not in self.tasks[predecessor_id].successors:
            self.tasks[predecessor_id].successors.append(successor_id)
        
        # Add to successor's predecessors
        if predecessor_id not in self.tasks[successor_id].predecessors:
            self.tasks[successor_id].predecessors.append(predecessor_id)
        
        print(f"[SCHEDULE] Added dependency: {predecessor_id} → {successor_id}")
        return True
    
    def update_task_duration(self, task_id: str, duration_days: int) -> bool:
        """Update the duration of an existing task."""
        if task_id not in self.tasks:
            print(f"[SCHEDULE] Error: Task {task_id} not found")
            return False
        
        old_duration = self.tasks[task_id].duration_days
        self.tasks[task_id].duration_days = duration_days
        
        print(f"[SCHEDULE] Updated task {task_id} duration: {old_duration} → {duration_days} days")
        return True
    
    def add_resource_requirement(self, task_id: str, resource: str) -> bool:
        """Add a resource requirement to a task."""
        if task_id not in self.tasks:
            print(f"[SCHEDULE] Error: Task {task_id} not found")
            return False
        
        if resource not in self.tasks[task_id].resource_requirements:
            self.tasks[task_id].resource_requirements.append(resource)
            print(f"[SCHEDULE] Added resource requirement: {resource} for task {task_id}")
        
        return True
    
    def get_schedule_summary(self) -> str:
        """Get a formatted summary of the current schedule."""
        if not self.tasks:
            return "No tasks in schedule yet."
        
        summary = f"Project: {self.project_name}\n"
        summary += f"Total Tasks: {len(self.tasks)}\n\n"
        
        for task_id, task in self.tasks.items():
            summary += f"Task: {task_id} - {task.name}\n"
            summary += f"  Duration: {task.duration_days} days\n"
            
            if task.predecessors:
                summary += f"  Predecessors: {', '.join(task.predecessors)}\n"
            
            if task.resource_requirements:
                summary += f"  Resources: {', '.join(task.resource_requirements)}\n"
            
            summary += f"  Description: {task.description}\n\n"
        
        return summary
    
    def export_schedule(self, filename: str) -> bool:
        """Export the schedule to a JSON file."""
        try:
            schedule_data = {
                "project_name": self.project_name,
                "project_start_date": self.project_start_date,
                "tasks": {}
            }
            
            for task_id, task in self.tasks.items():
                schedule_data["tasks"][task_id] = {
                    "name": task.name,
                    "description": task.description,
                    "duration_days": task.duration_days,
                    "predecessors": task.predecessors,
                    "successors": task.successors,
                    "resource_requirements": task.resource_requirements
                }
            
            with open(filename, 'w') as f:
                json.dump(schedule_data, f, indent=2)
            
            print(f"[SCHEDULE] Schedule exported to {filename}")
            return True
            
        except Exception as e:
            print(f"[SCHEDULE] Error exporting schedule: {e}")
            return False

# =============================================================================
# 4. CUSTOM LLM WRAPPER WITH KEY ROTATION
# =============================================================================

class Llama70BWithRotation(LLM):
    """
    Custom LLM wrapper for Cerebras llama-3.3-70b with automatic API key rotation.
    
    This class integrates with LangChain and automatically handles rate limiting
    by rotating through multiple API keys when 429 errors are encountered.
    """
    
    key_rotator: CerebrasAPIKeyRotator = None
    api_url: str = "https://api.cerebras.ai/v1/chat/completions"
    model_name: str = "llama-3.3-70b"
    max_retries: int = 10
    
    def __init__(self, key_rotator: CerebrasAPIKeyRotator, **kwargs):
        """
        Initialize the LLM with a key rotator.
        
        Args:
            key_rotator: Instance of CerebrasAPIKeyRotator
        """
        super().__init__(
            key_rotator=key_rotator,
            max_retries=len(key_rotator.api_keys),
            **kwargs
        )
        
        print(f"[LLM] Initialized Llama70B with {self.max_retries} API keys")
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type for LangChain."""
        return "cerebras_llama_70b"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Make an API call to Cerebras with automatic key rotation on rate limits.
        
        Args:
            prompt: The input prompt for the model
            stop: Optional stop sequences
            
        Returns:
            The model's response text
        """
        for attempt in range(self.max_retries):
            try:
                current_key = self.key_rotator.get_key()
                
                headers = {
                    "Authorization": f"Bearer {current_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
                
                if stop:
                    payload["stop"] = stop
                
                # Make the API request with timeout
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    print(f"[LLM] Rate limit hit (attempt {attempt + 1}/{self.max_retries})")
                    
                    if attempt < self.max_retries - 1:
                        if self.key_rotator.rotate_key():
                            time.sleep(2)  # Brief pause before retry
                            continue
                        else:
                            print("[LLM] All API keys exhausted")
                            break
                    else:
                        raise Exception("All API keys rate limited")
                
                # Check for other HTTP errors
                response.raise_for_status()
                
                # Parse the response
                response_data = response.json()
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    return response_data["choices"][0]["message"]["content"].strip()
                else:
                    raise Exception("Invalid response format from API")
                
            except requests.exceptions.RequestException as e:
                print(f"[LLM] Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise Exception(f"Failed to get response after {self.max_retries} attempts: {e}")
            
            except Exception as e:
                print(f"[LLM] Unexpected error: {e}")
                raise
        
        raise Exception("Max retries exceeded")

# =============================================================================
# 5. AGENT TOOLS DEFINITION
# =============================================================================

# Global instances (will be initialized in main)
doc_processor = None
schedule_state = None

# =============================================================================
# TOOL SCHEMAS FOR STRUCTURED INPUT
# =============================================================================

class SearchDocumentsSchema(BaseModel):
    query: str = Field(description="The search query describing what information you're looking for")

class AddTaskSchema(BaseModel):
    task_id: str = Field(description="Unique identifier for the task (e.g., 'TASK_001', 'FOUNDATION', etc.)")
    name: str = Field(description="Short, descriptive name for the task")
    description: str = Field(description="Detailed description of what the task involves")
    duration_days: int = Field(description="Estimated duration in working days")

class AddDependencySchema(BaseModel):
    successor_task_id: str = Field(description="ID of the task that depends on the predecessor")
    predecessor_task_id: str = Field(description="ID of the task that must be completed first")

class UpdateTaskDurationSchema(BaseModel):
    task_id: str = Field(description="ID of the task to update")
    duration_days: int = Field(description="New duration in working days")

class AddResourceRequirementSchema(BaseModel):
    task_id: str = Field(description="ID of the task")
    resource: str = Field(description="Description of the required resource")

class SetProjectInfoSchema(BaseModel):
    project_name: str = Field(description="Name of the construction project")
    start_date: str = Field(default=None, description="Project start date in YYYY-MM-DD format (optional)")

@tool  
def search_documents(query: str) -> str:
    """
    Search the construction documents for relevant information.
    
    Use this tool to find specific information from the uploaded construction documents
    such as project requirements, specifications, timelines, or any other details.
    
    Args:
        query: The search query describing what information you're looking for
        
    Returns:
        Relevant excerpts from the construction documents
    """
    if not doc_processor or not doc_processor.vectorstore:
        return "Error: Documents have not been processed yet. Please ensure documents are loaded first."
    
    results = doc_processor.search_documents(query, k=3)
    
    if not results:
        return f"No relevant information found for query: {query}"
    
    formatted_response = f"Found {len(results)} relevant excerpts for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        formatted_response += f"--- Excerpt {i} (from {result['metadata']['source']}) ---\n"
        formatted_response += result['content']
        formatted_response += "\n\n"
    
    return formatted_response

@tool
def add_task(input_string: str) -> str:
    """
    Add a new task to the construction schedule.
    
    Input format: JSON string with keys: task_id, name, description, duration_days
    Example: {"task_id": "TASK_001", "name": "Site Prep", "description": "Clear site", "duration_days": 5}
    """
    if not schedule_state:
        return "Error: Schedule state not initialized."
    
    try:
        import json
        # Try to parse as JSON
        data = json.loads(input_string)
        
        task_id = data.get("task_id")
        name = data.get("name") 
        description = data.get("description")
        duration_days = int(data.get("duration_days", 0))
        
        if not all([task_id, name, description, duration_days]):
            return "Error: Missing required fields. Need task_id, name, description, duration_days"
        
        success = schedule_state.add_task(task_id, name, description, duration_days)
        
        if success:
            return f"Successfully added task: {task_id} - {name} ({duration_days} days)"
        else:
            return f"Failed to add task: {task_id}"
            
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. Please use proper JSON syntax. Error: {e}"
    except Exception as e:
        return f"Error processing task: {e}"

@tool
def add_dependency(input_string: str) -> str:
    """
    Add a dependency relationship between two tasks.
    
    Input format: JSON string with keys: successor_task_id, predecessor_task_id
    Example: {"successor_task_id": "TASK_002", "predecessor_task_id": "TASK_001"}
    
    This establishes that the predecessor task must be completed before
    the successor task can begin.
    """
    if not schedule_state:
        return "Error: Schedule state not initialized."
    
    try:
        import json
        # Try to parse as JSON
        data = json.loads(input_string)
        
        successor_task_id = data.get("successor_task_id")
        predecessor_task_id = data.get("predecessor_task_id")
        
        if not all([successor_task_id, predecessor_task_id]):
            return "Error: Missing required fields. Need successor_task_id and predecessor_task_id"
        
        success = schedule_state.add_dependency(successor_task_id, predecessor_task_id)
        
        if success:
            return f"Successfully added dependency: {predecessor_task_id} must complete before {successor_task_id}"
        else:
            return f"Failed to add dependency between {predecessor_task_id} and {successor_task_id}"
            
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. Please use proper JSON syntax. Error: {e}"
    except Exception as e:
        return f"Error processing dependency: {e}"

@tool
def update_task_duration(task_id: str, duration_days: int) -> str:
    """
    Update the duration of an existing task.
    
    Args:
        task_id: ID of the task to update
        duration_days: New duration in working days
        
    Returns:
        Confirmation message about the duration update
    """
    if not schedule_state:
        return "Error: Schedule state not initialized."
    
    success = schedule_state.update_task_duration(task_id, duration_days)
    
    if success:
        return f"Successfully updated duration for task {task_id} to {duration_days} days"
    else:
        return f"Failed to update duration for task {task_id}"

@tool
def add_resource_requirement(task_id: str, resource: str) -> str:
    """
    Add a resource requirement to a task.
    
    Args:
        task_id: ID of the task
        resource: Description of the required resource (e.g., "Excavator", "Concrete crew", "Steel beams")
        
    Returns:
        Confirmation message about the resource requirement addition
    """
    if not schedule_state:
        return "Error: Schedule state not initialized."
    
    success = schedule_state.add_resource_requirement(task_id, resource)
    
    if success:
        return f"Successfully added resource requirement '{resource}' to task {task_id}"
    else:
        return f"Failed to add resource requirement to task {task_id}"

@tool
def get_current_schedule() -> str:
    """
    Get the current state of the construction schedule.
    
    Use this tool to review the tasks and dependencies you've added so far,
    which helps in planning additional tasks and ensuring completeness.
    
    Returns:
        A formatted summary of the current schedule including all tasks and dependencies
    """
    if not schedule_state:
        return "Error: Schedule state not initialized."
    
    return schedule_state.get_schedule_summary()

@tool
def set_project_info(project_name: str, start_date: str = None) -> str:
    """
    Set basic project information.
    
    Args:
        project_name: Name of the construction project
        start_date: Project start date in YYYY-MM-DD format (optional)
        
    Returns:
        Confirmation message
    """
    if not schedule_state:
        return "Error: Schedule state not initialized."
    
    schedule_state.set_project_info(project_name, start_date)
    return f"Set project name to '{project_name}'" + (f" with start date {start_date}" if start_date else "")

@tool
def finish_schedule() -> str:
    """
    Finalize and export the construction schedule with each task in a separate folder.
    
    This tool creates a structured directory with:
    - Main schedule file
    - Individual task folders with detailed task information
    - Dependencies and timeline information
    
    Returns:
        Confirmation message about the export
    """
    if not schedule_state:
        return "Error: Schedule state not initialized."
    
    if not schedule_state.tasks:
        return "Error: No tasks in schedule. Cannot export empty schedule."
    
    try:
        import os
        import json
        from datetime import datetime
        
        # Create main output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_dir = f"construction_schedule_{timestamp}"
        os.makedirs(main_dir, exist_ok=True)
        
        # Export main schedule file
        main_schedule_file = os.path.join(main_dir, "main_schedule.json")
        schedule_state.export_schedule(main_schedule_file)
        
        # Create individual task folders
        tasks_created = 0
        for task_id, task in schedule_state.tasks.items():
            # Create folder for this task
            task_folder = os.path.join(main_dir, f"Task_{task_id}")
            os.makedirs(task_folder, exist_ok=True)
            
            # Create detailed task file
            task_details = {
                "task_id": task.id,
                "name": task.name,
                "description": task.description,
                "duration_days": task.duration_days,
                "predecessors": task.predecessors,
                "successors": task.successors,
                "start_date": task.start_date,
                "end_date": task.end_date,
                "resource_requirements": task.resource_requirements,
                "created_timestamp": timestamp
            }
            
            # Save task details
            task_file = os.path.join(task_folder, f"{task_id}_details.json")
            with open(task_file, 'w') as f:
                json.dump(task_details, f, indent=2)
            
            # Create task summary text file
            summary_file = os.path.join(task_folder, f"{task_id}_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"CONSTRUCTION TASK SUMMARY\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Task ID: {task.id}\n")
                f.write(f"Name: {task.name}\n")
                f.write(f"Duration: {task.duration_days} days\n\n")
                f.write(f"Description:\n{task.description}\n\n")
                
                if task.predecessors:
                    f.write(f"Prerequisites (must complete first):\n")
                    for pred in task.predecessors:
                        pred_task = schedule_state.tasks.get(pred)
                        if pred_task:
                            f.write(f"  - {pred}: {pred_task.name}\n")
                    f.write("\n")
                
                if task.successors:
                    f.write(f"Following tasks (depend on this):\n")
                    for succ in task.successors:
                        succ_task = schedule_state.tasks.get(succ)
                        if succ_task:
                            f.write(f"  - {succ}: {succ_task.name}\n")
                    f.write("\n")
                
                if task.resource_requirements:
                    f.write(f"Required Resources:\n")
                    for resource in task.resource_requirements:
                        f.write(f"  - {resource}\n")
                    f.write("\n")
            
            tasks_created += 1
        
        # Create project overview file
        overview_file = os.path.join(main_dir, "project_overview.txt")
        with open(overview_file, 'w') as f:
            f.write(f"CONSTRUCTION PROJECT OVERVIEW\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Project: {schedule_state.project_name}\n")
            f.write(f"Total Tasks: {len(schedule_state.tasks)}\n")
            f.write(f"Export Date: {timestamp}\n\n")
            f.write(f"TASK SUMMARY:\n")
            f.write("-" * 30 + "\n")
            
            for task_id, task in schedule_state.tasks.items():
                f.write(f"{task_id}: {task.name} ({task.duration_days} days)\n")
        
        summary = schedule_state.get_schedule_summary()
        return f"""Successfully created structured schedule export in '{main_dir}/'!

Directory structure created:
- {main_dir}/
  - main_schedule.json (complete schedule data)
  - project_overview.txt (project summary)
  - Task_[ID]/ (individual folders for {tasks_created} tasks)
    - [TASK_ID]_details.json (detailed task data)
    - [TASK_ID]_summary.txt (human-readable summary)

Final Schedule Summary:
{summary}"""
        
    except Exception as e:
        return f"Error creating structured export: {e}"

# =============================================================================
# 6. AGENT ASSEMBLY & EXECUTION
# =============================================================================

def create_construction_agent(llm: Llama70BWithRotation) -> AgentExecutor:
    """
    Create the ReAct agent with all tools and system prompt.
    
    Args:
        llm: The custom LLM instance with key rotation
        
    Returns:
        Configured agent executor
    """
    
    # Define the tools available to the agent
    tools = [
        search_documents,
        add_task,
        add_dependency, 
        update_task_duration,
        add_resource_requirement,
        get_current_schedule,
        set_project_info,
        finish_schedule
    ]
    
    # Create the system prompt for the ReAct agent
    system_prompt = """You are an expert AI Construction Scheduling Agent. Your goal is to analyze construction documents and create a comprehensive, realistic project schedule.

INSTRUCTIONS:
1. First, search the documents to understand the project scope, requirements, and any timeline constraints.
2. Set the project information using the project name you identify from the documents.
3. Identify all major construction tasks from the documents (foundation, framing, electrical, plumbing, etc.).
4. For each task, determine:
   - Realistic duration estimates based on industry standards and document specifications
   - Dependencies between tasks (what must be completed before each task can start)
   - Required resources (equipment, materials, crew types)
5. Build the schedule incrementally, adding tasks and their dependencies systematically.
6. Review your work periodically using get_current_schedule to ensure completeness.
7. When finished, export the final schedule.

REASONING STRATEGY:
- Always think step-by-step and explain your reasoning
- Use the search_documents tool extensively to base decisions on actual document content
- Consider typical construction sequences and dependencies
- Be realistic about timeframes - construction takes time!
- Include all phases: site preparation, foundation, structure, systems, finishes

CRITICAL REMINDERS:
- You must use the available tools - do not try to create schedules without using add_task
- Search documents first before making assumptions
- Add dependencies carefully - most tasks have prerequisites
- Use get_current_schedule regularly to track your progress

Begin by searching the documents to understand the project scope and requirements."""
    
    # Create the ReAct prompt template
    react_prompt = PromptTemplate.from_template(
        system_prompt + """

{tools}

Use the following format:

Thought: I need to think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: {{"parameter_name": "value", "parameter_name2": value}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final response to the user

IMPORTANT: For Action Input format:
- search_documents: Just provide the search query as plain text
- add_task: Provide JSON string like {{"task_id": "TASK_001", "name": "Site Preparation", "description": "Clear and excavate site", "duration_days": 5}}
- add_dependency: Provide JSON string like {{"successor_task_id": "TASK_002", "predecessor_task_id": "TASK_001"}}
- Other tools: Use the parameter values directly

Question: {input}
{agent_scratchpad}"""
    )
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    
    # Create agent executor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=50,
        max_execution_time=1800  # 30 minutes timeout
    )
    
    return agent_executor

def main():
    """
    Main function to configure and run the AI Construction Scheduling Agent.
    """
    print("=" * 70)
    print("AI CONSTRUCTION SCHEDULING AGENT")
    print("=" * 70)
    
    # =============================================================================
    # CONFIGURATION SECTION
    # =============================================================================
    print("\n[CONFIG] Starting configuration...")
    
    # Get Cerebras API keys from environment or user input
    api_keys = []
    
    # Try to load from environment first
    env_keys = [
        os.environ.get('CEREBRAS_API_KEY_1'),
        os.environ.get('CEREBRAS_API_KEY_2'),
        os.environ.get('CEREBRAS_API_KEY_3'),
        os.environ.get('CEREBRAS_API_KEY_4'),
        os.environ.get('CEREBRAS_API_KEY_5')
    ]
    
    api_keys = [key for key in env_keys if key]
    
    if not api_keys:
        print("\n[CONFIG] No API keys found in environment variables.")
        print("Please provide your Cerebras API keys.")
        print("You can provide multiple keys for automatic rotation when rate limits are hit.")
        print("Press Enter with empty input when done.")
        
        key_count = 1
        while True:
            key = input(f"API Key {key_count} (or Enter to finish): ").strip()
            if not key:
                break
            api_keys.append(key)
            key_count += 1
    
    if not api_keys:
        print("[ERROR] No API keys provided. Cannot proceed without at least one API key.")
        print("Set environment variables CEREBRAS_API_KEY_1, CEREBRAS_API_KEY_2, etc., or provide them when prompted.")
        return
    
    # Get document paths
    document_paths = []
    
    print(f"\n[CONFIG] Found {len(api_keys)} API keys.")
    print("Now please provide paths to your construction documents (PDF files).")
    print("Press Enter with empty input when done.")
    
    doc_count = 1
    while True:
        path = input(f"Document {doc_count} path (or Enter to finish): ").strip()
        if not path:
            break
        
        if os.path.exists(path) and path.lower().endswith('.pdf'):
            document_paths.append(path)
            doc_count += 1
        else:
            print(f"[WARNING] File not found or not a PDF: {path}")
    
    if not document_paths:
        print("[ERROR] No valid document paths provided. Cannot proceed without construction documents.")
        return
    
    print(f"\n[CONFIG] Configuration complete:")
    print(f"  - API Keys: {len(api_keys)}")
    print(f"  - Documents: {len(document_paths)}")
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    try:
        # Initialize global components
        global doc_processor, schedule_state
        
        print("\n[INIT] Initializing components...")
        
        # Initialize key rotator
        key_rotator = CerebrasAPIKeyRotator(api_keys)
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Initialize schedule state
        schedule_state = ScheduleState()
        
        # Initialize custom LLM
        llm = Llama70BWithRotation(key_rotator)
        
        print("[INIT] All components initialized successfully")
        
        # =============================================================================
        # DOCUMENT PROCESSING
        # =============================================================================
        
        print("\n[PROCESSING] Processing construction documents...")
        
        if not doc_processor.process_documents(document_paths):
            print("[ERROR] Failed to process documents. Cannot proceed.")
            return
        
        print("[PROCESSING] Document processing completed successfully")
        
        # =============================================================================
        # AGENT EXECUTION
        # =============================================================================
        
        print("\n[AGENT] Creating and starting the construction scheduling agent...")
        
        agent_executor = create_construction_agent(llm)
        
        # Start the agent with the main task
        task_description = f"""
        Please analyze the {len(document_paths)} construction documents that have been loaded and create a comprehensive project schedule.
        
        The documents available are:
        {', '.join([os.path.basename(path) for path in document_paths])}
        
        Your task is to:
        1. Understand the project scope and requirements from the documents
        2. Identify all major construction tasks and activities
        3. Estimate realistic durations for each task
        4. Determine task dependencies and sequencing
        5. Create a complete, actionable construction schedule
        6. Export the final schedule to a file
        
        Begin by searching the documents to understand what type of construction project this is.
        """
        
        print(f"[AGENT] Starting agent execution with task description...")
        print(f"[AGENT] Processing {len(document_paths)} documents...")
        
        # Execute the agent
        result = agent_executor.invoke({"input": task_description})
        
        print("\n" + "=" * 70)
        print("AGENT EXECUTION COMPLETED")
        print("=" * 70)
        print(f"Result: {result['output']}")
        
        # Print final schedule summary
        if schedule_state.tasks:
            print("\n" + "=" * 70)
            print("FINAL SCHEDULE SUMMARY")
            print("=" * 70)
            print(schedule_state.get_schedule_summary())
        
    except KeyboardInterrupt:
        print("\n[INFO] Agent execution interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 