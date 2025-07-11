import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime
import tempfile
from typing import List, Dict, Any
import json

# For RAG implementation
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# For document processing
try:
    import PyPDF2
    import docx
    import openpyxl
    import xlrd
    HAS_DOC_PROCESSORS = True
except ImportError:
    HAS_DOC_PROCESSORS = False

# For Gemini API
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Database setup
DB_PATH = 'database.db'

# Initialize the database schema
SCHEMA_SQL = '''
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS ticket_conversations;
DROP TABLE IF EXISTS ticket_comments;
DROP TABLE IF EXISTS maintenance_tickets;
DROP TABLE IF EXISTS complaint_tickets;
DROP TABLE IF EXISTS billing_tickets;
DROP TABLE IF EXISTS service_tickets;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS leases;
DROP TABLE IF EXISTS units;
DROP TABLE IF EXISTS agents;
DROP TABLE IF EXISTS properties;
DROP TABLE IF EXISTS tenants;

CREATE TABLE tenants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    first_name      TEXT    NOT NULL,
    last_name       TEXT    NOT NULL,
    email           TEXT    UNIQUE NOT NULL,
    phone           TEXT,
    date_of_birth   TEXT,
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE properties (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    name            TEXT    NOT NULL,
    address_line1   TEXT    NOT NULL,
    address_line2   TEXT,
    city            TEXT    NOT NULL,
    state           TEXT,
    postal_code     TEXT,
    country         TEXT    NOT NULL,
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE units (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    unit_number     TEXT    NOT NULL,
    floor           TEXT,
    bedrooms        INTEGER,
    bathrooms       REAL,
    square_feet     INTEGER,
    status          TEXT    DEFAULT 'available',
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE leases (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    tenant_id       INTEGER NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    unit_id         INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
    start_date      DATETIME NOT NULL,
    end_date        DATETIME NOT NULL,
    rent_amount     REAL    NOT NULL,
    security_deposit REAL,
    status          TEXT    DEFAULT 'active',
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE agents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    first_name      TEXT,
    last_name       TEXT,
    role            TEXT,
    email           TEXT    UNIQUE,
    phone           TEXT,
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE service_tickets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    lease_id        INTEGER NOT NULL REFERENCES leases(id) ON DELETE CASCADE,
    raised_by       INTEGER        REFERENCES tenants(id) ON DELETE SET NULL,
    assigned_to     INTEGER        REFERENCES agents(id) ON DELETE SET NULL,
    category        TEXT     NOT NULL,
    description     TEXT     NOT NULL,
    status          TEXT     DEFAULT 'open',
    priority        TEXT     DEFAULT 'normal',
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    updated_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE maintenance_tickets (
    ticket_id       INTEGER PRIMARY KEY REFERENCES service_tickets(id) ON DELETE CASCADE,
    subcategory     TEXT,
    scheduled_for   DATETIME,
    technician_id   INTEGER REFERENCES agents(id) ON DELETE SET NULL
);

CREATE TABLE complaint_tickets (
    ticket_id       INTEGER PRIMARY KEY REFERENCES service_tickets(id) ON DELETE CASCADE,
    severity        TEXT    NOT NULL,
    complaint_type  TEXT,
    resolved_on     DATETIME
);

CREATE TABLE billing_tickets (
    ticket_id       INTEGER PRIMARY KEY REFERENCES service_tickets(id) ON DELETE CASCADE,
    invoice_number  TEXT    NOT NULL,
    amount_disputed REAL,
    resolution_date DATETIME
);

CREATE TABLE ticket_comments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    ticket_id       INTEGER NOT NULL REFERENCES service_tickets(id) ON DELETE CASCADE,
    author_id       INTEGER NOT NULL,
    author_type     TEXT    NOT NULL,
    comment_text    TEXT    NOT NULL,
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE ticket_conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    ticket_id       INTEGER NOT NULL REFERENCES service_tickets(id) ON DELETE CASCADE,
    author_type     TEXT    NOT NULL,
    author_id       INTEGER NOT NULL,
    message_text    TEXT    NOT NULL,
    sent_at         DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);

CREATE TABLE payments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours')),
    lease_id        INTEGER NOT NULL REFERENCES leases(id) ON DELETE CASCADE,
    payment_type    TEXT    NOT NULL,
    billing_period  TEXT,
    due_date        DATETIME,
    amount          REAL    NOT NULL,
    method          TEXT,
    paid_on         DATETIME,
    reference_number TEXT,
    created_at      DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','+8 hours'))
);
'''

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        try:
            with self.get_connection() as conn:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
                return df
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return None

    def get_table_info(self):
        """Get information about all tables"""
        query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        return self.execute_query(query)

    def get_table_schema(self, table_name: str):
        """Get schema for a specific table"""
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file):
        """Extract text from PDF file"""
        if not HAS_DOC_PROCESSORS:
            return "PDF processing not available. Install PyPDF2."

        try:
            # Reset file pointer
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            if len(pdf_reader.pages) == 0:
                return "Error: PDF file appears to be empty or corrupted."
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"Page {i+1}:\n{page_text}\n\n"
                except Exception as page_error:
                    # Continue processing other pages if one fails
                    text += f"Page {i+1}: Error extracting text - {str(page_error)}\n\n"
                    continue
            
            if not text.strip():
                return "Warning: No readable text found in PDF. The PDF might contain only images or be password protected."
            
            return text
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    @staticmethod
    def extract_text_from_docx(file):
        """Extract text from DOCX file"""
        if not HAS_DOC_PROCESSORS:
            return "DOCX processing not available. Install python-docx."

        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error processing DOCX: {str(e)}"

    @staticmethod
    def extract_text_from_txt(file):
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            return f"Error processing TXT: {str(e)}"

    @staticmethod
    def extract_text_from_csv(file):
        """Extract text from CSV file"""
        try:
            # Reset file pointer
            file.seek(0)
            df = pd.read_csv(file)
            
            # Convert DataFrame to meaningful text
            text = f"CSV Data Summary:\n"
            text += f"Total rows: {len(df)}\n"
            text += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            
            # Add column information
            text += "Column Details:\n"
            for col in df.columns:
                text += f"- {col}: {df[col].dtype}\n"
            
            text += "\nData Sample:\n"
            # Convert first few rows to text
            for idx, row in df.head(10).iterrows():
                text += f"Row {idx + 1}: "
                for col, value in row.items():
                    text += f"{col}={value}, "
                text += "\n"
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                text += "\nNumeric Summary:\n"
                for col in numeric_cols:
                    text += f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
            
            return text
        except Exception as e:
            return f"Error processing CSV: {str(e)}"

    @staticmethod
    def extract_text_from_excel(file):
        """Extract text from Excel file"""
        if not HAS_DOC_PROCESSORS:
            return "Excel processing not available. Install openpyxl and xlrd."

        try:
            # Reset file pointer
            file.seek(0)
            
            # Try to read as Excel file
            try:
                # For .xlsx files
                df_dict = pd.read_excel(file, sheet_name=None, engine='openpyxl')
            except:
                try:
                    # For .xls files
                    file.seek(0)
                    df_dict = pd.read_excel(file, sheet_name=None, engine='xlrd')
                except:
                    return "Error: Could not read Excel file. Ensure it's a valid .xlsx or .xls file."
            
            text = f"Excel File Analysis:\n"
            text += f"Number of sheets: {len(df_dict)}\n\n"
            
            # Process each sheet
            for sheet_name, df in df_dict.items():
                text += f"Sheet: {sheet_name}\n"
                text += f"  Rows: {len(df)}, Columns: {len(df.columns)}\n"
                text += f"  Column names: {', '.join(df.columns.astype(str).tolist())}\n"
                
                # Add data sample for each sheet
                if len(df) > 0:
                    text += f"  Sample data from {sheet_name}:\n"
                    for idx, row in df.head(5).iterrows():
                        text += f"    Row {idx + 1}: "
                        for col, value in row.items():
                            text += f"{col}={value}, "
                        text += "\n"
                
                # Add summary for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    text += f"  Numeric summary for {sheet_name}:\n"
                    for col in numeric_cols:
                        if not df[col].isna().all():
                            text += f"    {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
                
                text += "\n"
            
            return text
        except Exception as e:
            return f"Error processing Excel file: {str(e)}"

class RAGSystem:
    def __init__(self):
        self.embeddings_model = None
        self.index = None
        self.documents = []
        self.document_embeddings = []

    def initialize_embeddings(self):
        """Initialize the embeddings model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            st.error("Sentence Transformers not available. Install sentence-transformers for RAG functionality.")
            return False

        try:
            with st.spinner("Loading embeddings model... This may take a moment on first run."):
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("Embeddings model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error initializing embeddings model: {str(e)}")
            return False

    def add_documents(self, documents: List[str]):
        """Add documents to the RAG system"""
        if not self.embeddings_model:
            st.error("Embeddings model not initialized. Please initialize first.")
            return False

        try:
            with st.spinner("Processing documents and generating embeddings..."):
                # Filter out empty documents
                valid_documents = [doc for doc in documents if doc and doc.strip()]
                
                if not valid_documents:
                    st.warning("No valid text found in documents.")
                    return False

                # Split documents into chunks
                chunks = []
                for doc in valid_documents:
                    doc_chunks = self._split_text(doc)
                    chunks.extend(doc_chunks)

                if not chunks:
                    st.warning("No text chunks created from documents.")
                    return False

                # Generate embeddings with progress
                st.info(f"Generating embeddings for {len(chunks)} text chunks...")
                embeddings = self.embeddings_model.encode(chunks, show_progress_bar=False)

                # Create or update FAISS index
                if self.index is None:
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    st.info(f"Created new FAISS index with dimension {dimension}")

                # Add embeddings to index
                self.index.add(embeddings.astype(np.float32))
                self.documents.extend(chunks)
                self.document_embeddings.extend(embeddings)

                st.success(f"Successfully added {len(chunks)} text chunks to the knowledge base!")
                return True
                
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            st.exception(e)  # This will show the full stack trace
            return False

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def retrieve_relevant_docs(self, query: str, k: int = 3):
        """Retrieve relevant documents for a query"""
        if not self.embeddings_model or not self.index:
            return []

        try:
            if self.index.ntotal == 0:  # Check if index is empty
                return []
                
            query_embedding = self.embeddings_model.encode([query])
            distances, indices = self.index.search(query_embedding.astype(np.float32), min(k, self.index.ntotal))

            relevant_docs = []
            for idx in indices[0]:
                if idx < len(self.documents) and idx >= 0:
                    relevant_docs.append(self.documents[idx])

            return relevant_docs
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []

class ChatBot:
    def __init__(self, db_manager: DatabaseManager, rag_system: RAGSystem):
        self.db_manager = db_manager
        self.rag_system = rag_system
        self.gemini_model = None

    def set_gemini_api_key(self, api_key: str):
        """Set Gemini API key"""
        if not HAS_GEMINI:
            st.error("Google GenerativeAI library not available. Install google-generativeai package.")
            return False

        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            return True
        except Exception as e:
            st.error(f"Error setting Gemini API key: {str(e)}")
            return False

    def generate_sql_query(self, user_question: str) -> str:
        """Generate SQL query based on user question"""
        schema_info = self._get_database_schema()

        prompt = f"""
        You are a SQL expert. Based on the following database schema and user question,
        generate a SQL query that would answer the question.

        Database Schema:
        {schema_info}

        User Question: {user_question}

        Please provide only the SQL query without any explanation. Make sure the SQL is valid SQLite syntax.
        """

        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                sql_query = response.text.strip()

                # Clean up the response to extract only SQL
                if "```sql" in sql_query:
                    sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
                elif "```" in sql_query:
                    sql_query = sql_query.split("```")[1].split("```")[0].strip()

                return sql_query
            except Exception as e:
                st.error(f"Error generating SQL query with Gemini: {str(e)}")
                return self._generate_simple_query(user_question)
        else:
            # Fallback to simple query generation
            return self._generate_simple_query(user_question)

    def _get_database_schema(self) -> str:
        """Get database schema information"""
        tables = self.db_manager.get_table_info()
        schema_info = "Database Tables:\n"

        if tables is not None:
            for _, row in tables.iterrows():
                table_name = row['name']
                schema_info += f"\nTable: {table_name}\n"
                table_schema = self.db_manager.get_table_schema(table_name)
                if table_schema is not None:
                    for _, col in table_schema.iterrows():
                        schema_info += f"  - {col['name']} ({col['type']})\n"

        return schema_info

    def _generate_simple_query(self, question: str) -> str:
        """Generate simple SQL queries based on keywords"""
        question_lower = question.lower()

        if "tenant" in question_lower:
            return "SELECT * FROM tenants LIMIT 10"
        elif "property" in question_lower or "properties" in question_lower:
            return "SELECT * FROM properties LIMIT 10"
        elif "lease" in question_lower:
            return "SELECT * FROM leases LIMIT 10"
        elif "ticket" in question_lower:
            return "SELECT * FROM service_tickets LIMIT 10"
        elif "payment" in question_lower:
            return "SELECT * FROM payments LIMIT 10"
        else:
            return "SELECT name FROM sqlite_master WHERE type='table'"

    def answer_question(self, question: str) -> str:
        """Answer user question using RAG and database query"""
        response = f"**Question:** {question}\n\n"

        # Try to get relevant documents from RAG
        relevant_docs = self.rag_system.retrieve_relevant_docs(question)

        # Try to generate and execute SQL query
        sql_query = self.generate_sql_query(question)

        if sql_query:
            response += f"**Generated SQL Query:**\n```sql\n{sql_query}\n```\n\n"

            # Execute the query
            df = self.db_manager.execute_query(sql_query)
            if df is not None and not df.empty:
                response += f"**Database Results:**\n"
                response += df.to_string(index=False)
                response += "\n\n"
            else:
                response += "**Database Results:** No data found or query error.\n\n"

        # Add relevant documents if available
        if relevant_docs:
            response += f"**Relevant Documents:**\n"
            for i, doc in enumerate(relevant_docs[:2], 1):
                response += f"{i}. {doc[:200]}...\n\n"

        # Generate AI response if Gemini is available
        if self.gemini_model:
            context = f"Database query result: {df.to_string() if df is not None else 'No results'}\n"
            context += f"Relevant documents: {' '.join(relevant_docs[:2])}\n"

            ai_prompt = f"""
            You are a helpful assistant for a property management system. Based on the following context,
            provide a clear, concise, and helpful answer to the user's question.

            Question: {question}

            Context:
            {context}

            Please provide a comprehensive answer that interprets the data and gives practical insights.
            Focus on being helpful and actionable in your response.
            """

            try:
                ai_response = self.gemini_model.generate_content(ai_prompt)
                response += f"**AI Response:**\n{ai_response.text}"
            except Exception as e:
                response += f"**AI Response Error:** {str(e)}"

        return response

# Streamlit App
def main():
    st.set_page_config(
        page_title="Property Management RAG Chatbot",
        page_icon="🏠",
        layout="wide"
    )

    # Main app starts here
    st.title("🏠 Property Management RAG Chatbot")
    st.markdown("Ask questions about your property management data and upload documents for enhanced context.")

    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager(DB_PATH)

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot(st.session_state.db_manager, st.session_state.rag_system)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'gemini_configured' not in st.session_state:
        st.session_state.gemini_configured = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Gemini API Key
        st.subheader("🔑 Gemini API Key")
        if not HAS_GEMINI:
            st.error("Please install google-generativeai: `pip install google-generativeai`")
        else:
            api_key = st.text_input("Enter your Gemini API Key", type="password")
            if st.button("Set API Key"):
                if api_key:
                    if st.session_state.chatbot.set_gemini_api_key(api_key):
                        st.session_state.gemini_configured = True
                        st.success("Gemini API Key configured successfully!")
                    else:
                        st.error("Failed to configure API Key")
                else:
                    st.error("Please enter an API Key")

            if st.session_state.gemini_configured:
                st.success("✅ Gemini API configured")

        # Document Upload
        st.subheader("📄 Document Upload")
        if not HAS_SENTENCE_TRANSFORMERS:
            st.error("Please install sentence-transformers: `pip install sentence-transformers`")

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, CSV, Excel)",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload relevant documents to enhance the chatbot's knowledge base"
        )

        if uploaded_files:
            if st.button("Process Documents"):
                try:
                    # Initialize embeddings model if not already done
                    if not st.session_state.rag_system.embeddings_model:
                        if not st.session_state.rag_system.initialize_embeddings():
                            st.error("Failed to initialize embeddings model. Please try again.")
                            st.stop()

                    documents = []
                    processing_success = True
                    
                    with st.spinner("Processing uploaded files..."):
                        for file in uploaded_files:
                            try:
                                st.info(f"Processing: {file.name}")
                                
                                if file.type == "application/pdf":
                                    text = DocumentProcessor.extract_text_from_pdf(file)
                                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                    text = DocumentProcessor.extract_text_from_docx(file)
                                elif file.type == "text/plain":
                                    text = DocumentProcessor.extract_text_from_txt(file)
                                elif file.type == "text/csv":
                                    text = DocumentProcessor.extract_text_from_csv(file)
                                elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                                   "application/vnd.ms-excel"]:
                                    text = DocumentProcessor.extract_text_from_excel(file)
                                else:
                                    st.warning(f"Unsupported file type: {file.type}")
                                    continue

                                # Check if text extraction was successful
                                if text and not text.startswith("Error"):
                                    documents.append(text)
                                    st.success(f"✅ Extracted text from: {file.name}")
                                else:
                                    st.warning(f"⚠️ Could not extract text from: {file.name} - {text}")
                                    processing_success = False

                            except Exception as file_error:
                                st.error(f"❌ Error processing {file.name}: {str(file_error)}")
                                processing_success = False
                                continue

                    # Add documents to RAG system
                    if documents:
                        if st.session_state.rag_system.add_documents(documents):
                            st.success(f"🎉 Successfully processed {len(documents)} documents and added to knowledge base!")
                        else:
                            st.error("Failed to add documents to knowledge base")
                    elif not processing_success:
                        st.error("No documents were successfully processed")
                    else:
                        st.warning("No valid documents found to process")
                        
                except Exception as e:
                    st.error(f"Critical error during document processing: {str(e)}")
                    st.exception(e)

        # Database Info
        st.subheader("🗄️ Database Info")
        if st.button("Show Tables"):
            tables = st.session_state.db_manager.get_table_info()
            if tables is not None:
                st.dataframe(tables)

        # Sample Data
        st.subheader("📝 Sample Data")
        if st.button("Insert Sample Data"):
            insert_sample_data()
            st.success("Sample data inserted!")

        # System Status
        st.subheader("🔍 System Status")
        with st.expander("View System Information"):
            st.write("**Embeddings Model Status:**", 
                    "✅ Loaded" if st.session_state.rag_system.embeddings_model else "❌ Not loaded")
            
            if st.session_state.rag_system.index:
                st.write("**Vector Index:**", f"✅ {st.session_state.rag_system.index.ntotal} documents indexed")
            else:
                st.write("**Vector Index:**", "❌ No index created")
            
            st.write("**Total Document Chunks:**", len(st.session_state.rag_system.documents))
            st.write("**Gemini API:**", "✅ Configured" if st.session_state.gemini_configured else "❌ Not configured")
            
            # Memory usage approximation
            import sys
            total_size = sum(sys.getsizeof(doc) for doc in st.session_state.rag_system.documents)
            st.write("**Approximate Memory Usage:**", f"{total_size / 1024 / 1024:.2f} MB")
            
            if st.button("Clear Document Cache"):
                st.session_state.rag_system.documents = []
                st.session_state.rag_system.document_embeddings = []
                st.session_state.rag_system.index = None
                st.success("Document cache cleared!")

    # Main chat interface
    st.header("💬 Chat Interface")

    # Instructions
    if not st.session_state.gemini_configured:
        st.info("👆 Please configure your Gemini API key in the sidebar to enable AI responses.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your property management data..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.answer_question(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def insert_sample_data():
    """Insert sample data into the database"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        # Sample tenants
        tenants_data = [
            ('John', 'Doe', 'john.doe@email.com', '+1-555-123-4567', '1990-01-15'),
            ('Jane', 'Smith', 'jane.smith@email.com', '+1-555-234-5678', '1985-03-22'),
            ('Bob', 'Johnson', 'bob.johnson@email.com', '+1-555-345-6789', '1992-07-08'),
            ('Alice', 'Brown', 'alice.brown@email.com', '+1-555-456-7890', '1988-11-30'),
            ('Charlie', 'Wilson', 'charlie.wilson@email.com', '+1-555-567-8901', '1995-05-12'),
        ]

        for tenant in tenants_data:
            c.execute("INSERT OR IGNORE INTO tenants (first_name, last_name, email, phone, date_of_birth) VALUES (?, ?, ?, ?, ?)", tenant)

        # Sample properties
        properties_data = [
            ('Sunset Apartments', '123 Main St', 'Unit A', 'Downtown', 'CA', '90210', 'USA'),
            ('Ocean View Complex', '456 Beach Blvd', 'Building B', 'Coastal', 'CA', '90211', 'USA'),
            ('Mountain Ridge', '789 Hill Road', 'Tower C', 'Uptown', 'CA', '90212', 'USA'),
        ]

        for prop in properties_data:
            c.execute("INSERT OR IGNORE INTO properties (name, address_line1, address_line2, city, state, postal_code, country) VALUES (?, ?, ?, ?, ?, ?, ?)", prop)

        # Sample units
        units_data = [
            (1, '101', '1', 2, 1.5, 850, 'occupied'),
            (1, '102', '1', 1, 1.0, 650, 'available'),
            (2, '201', '2', 3, 2.0, 1200, 'occupied'),
            (2, '202', '2', 2, 2.0, 1000, 'maintenance'),
            (3, '301', '3', 1, 1.0, 700, 'available'),
        ]

        for unit in units_data:
            c.execute("INSERT OR IGNORE INTO units (property_id, unit_number, floor, bedrooms, bathrooms, square_feet, status) VALUES (?, ?, ?, ?, ?, ?, ?)", unit)

        # Sample agents
        agents_data = [
            ('Mike', 'Manager', 'Property Manager', 'mike.manager@company.com', '+1-555-111-2222'),
            ('Sarah', 'Tech', 'Maintenance Technician', 'sarah.tech@company.com', '+1-555-333-4444'),
            ('David', 'Admin', 'Administrative Assistant', 'david.admin@company.com', '+1-555-555-6666'),
        ]

        for agent in agents_data:
            c.execute("INSERT OR IGNORE INTO agents (first_name, last_name, role, email, phone) VALUES (?, ?, ?, ?, ?)", agent)

        # Sample leases
        leases_data = [
            (1, 1, '2024-01-01', '2024-12-31', 1500.00, 1500.00, 'active'),
            (2, 3, '2024-02-01', '2025-01-31', 2200.00, 2200.00, 'active'),
            (3, 1, '2023-06-01', '2024-05-31', 1800.00, 1800.00, 'expired'),
        ]

        for lease in leases_data:
            c.execute("INSERT OR IGNORE INTO leases (tenant_id, unit_id, start_date, end_date, rent_amount, security_deposit, status) VALUES (?, ?, ?, ?, ?, ?, ?)", lease)

        # Sample service tickets
        tickets_data = [
            (1, 1, 1, 'maintenance', 'Leaking faucet in kitchen', 'open', 'normal'),
            (2, 2, 2, 'complaint', 'Noisy neighbors upstairs', 'in_progress', 'high'),
            (1, 1, 3, 'billing', 'Question about utility charges', 'closed', 'low'),
        ]

        for ticket in tickets_data:
            c.execute("INSERT OR IGNORE INTO service_tickets (lease_id, raised_by, assigned_to, category, description, status, priority) VALUES (?, ?, ?, ?, ?, ?, ?)", ticket)

        # Sample payments
        payments_data = [
            (1, 'rent', '2024-01', '2024-01-01', 1500.00, 'bank_transfer', '2024-01-01', 'TXN001'),
            (2, 'rent', '2024-02', '2024-02-01', 2200.00, 'check', '2024-02-02', 'CHK001'),
            (1, 'utilities', '2024-01', '2024-01-15', 150.00, 'credit_card', '2024-01-15', 'CC001'),
        ]

        for payment in payments_data:
            c.execute("INSERT OR IGNORE INTO payments (lease_id, payment_type, billing_period, due_date, amount, method, paid_on, reference_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", payment)

        conn.commit()

if __name__ == "__main__":
    main()
