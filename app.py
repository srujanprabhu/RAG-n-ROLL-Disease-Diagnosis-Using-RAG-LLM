from trulens.core import TruSession
import plotly.express as px
from datetime import datetime
import streamlit as st
import snowflake.connector as sf
from snowflake.snowpark import Session
import pandas as pd
from PyPDF2 import PdfReader
import re

st.set_page_config(
    page_title="Disease Diagnosis Using RAG-LLM",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

def record_feedback(question, response, context, response_time, session):
    """Record feedback for a query using TruLens"""
    try:
        # Create feedback record
        record = tru.record()
        record.add_input(question=question)
        record.add_output(response=response)
        record.add_metadata(
            context=context,
            response_time=response_time,
            timestamp=datetime.now()
        )
        st.session_state.metrics_history['timestamp'].append(datetime.now())
        st.session_state.metrics_history['question'].append(question)
        st.session_state.metrics_history['response'].append(response)
        st.session_state.metrics_history['context'].append(context)
        st.session_state.metrics_history['response_time'].append(response_time)
        feedback_prompt = f"""
        Analyze this interaction and provide a single score between 0 and 1 based on:
        1. Response relevance to question
        2. Response completeness
        3. Response accuracy based on context

        Question: {question}
        Context: {context}
        Response: {response}

        Return only a number between 0 and 1:
        """

        feedback_query = f"""
        SELECT snowflake.cortex.complete(
            'mistral-large',
            '{feedback_prompt.replace("'", "''")}'
        ) AS score
        """
        feedback_result = session.sql(feedback_query).to_pandas()
        feedback_score = float(feedback_result.iloc[0]['SCORE'].strip())
        st.session_state.metrics_history['feedback_scores'].append(feedback_score)
        record.add_metadata(feedback_score=feedback_score)
        return record
    except Exception as e:
        st.error(f"Error recording feedback: {str(e)}")
        return None
if 'first_run' not in st.session_state:
    st.session_state.first_run = True
# Connection parameters
sf_connection = sf.connect(
    user="SRUJANPRABHU",
    password="sp123@SP123",
    account="xcvpeuv-jn10394",
    warehouse="COMPUTE_WH",
    database="snowflake_llm_poc",
    schema="PUBLIC"
)
session = Session.builder.configs({
    "account": "xcvpeuv-jn10394",
    "user": "SRUJANPRABHU",
    "password": "sp123@SP123",
    "warehouse": "COMPUTE_WH",
    "database": "snowflake_llm_poc",
    "schema": "PUBLIC"
}).create()
if st.session_state.first_run:
    create_temp_table_query = """
    CREATE OR REPLACE TABLE temp (
        chunk_id INTEGER,
        chunk_text TEXT,
        text_vec VECTOR(FLOAT, 1024)
    )
    """
    session.sql(create_temp_table_query).collect()
    st.session_state.first_run = False
def clean_text(text):
    # Remove special characters and excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("'", "''")  # Escape single quotes for SQL
    text = text.strip()
    return text
def verify_temp_table():
    try:
        # Check if table exists
        table_check_query = """
        SELECT COUNT(*) as count 
        FROM information_schema.tables 
        WHERE table_name = 'TEMP'
        """
        table_exists = session.sql(table_check_query).collect()[0]['COUNT'] > 0
        if not table_exists:
            st.error("Temp table does not exist. Will create it now.")
            # Recreate the table because we have to cleaar the temp table for every rerun
            create_temp_table_query = """
            CREATE OR REPLACE TABLE temp (
                chunk_id INTEGER,
                chunk_text TEXT,
                text_vec VECTOR(FLOAT, 1024)
            )
            """
            session.sql(create_temp_table_query).collect()
            return "Table created successfully"
        count_query = "SELECT COUNT(*) as count FROM temp"
        record_count = session.sql(count_query).collect()[0]['COUNT']
        return f"Temp table exists with {record_count} records"
    except Exception as e:
        return f"Error verifying temp table: {str(e)}"
def process_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        total_text = ""
        for page in reader.pages:
            total_text += page.extract_text()

        # Clean the extracted text
        total_text = clean_text(total_text)
        if not total_text:
            return False, "No text could be extracted from the PDF"
        # Split into 10 chunks
        num_chunks = 10
        chunk_size = max(1, len(total_text) // num_chunks)
        chunks = [total_text[i:i + chunk_size] for i in range(0, len(total_text), chunk_size)][:num_chunks]

        # Get max chunk_id
        max_id_query = "SELECT COALESCE(MAX(chunk_id), 0) AS max_id FROM temp"
        max_id = session.sql(max_id_query).collect()[0]['MAX_ID']

        # Process chunks in batches for better optimization
        successful_inserts = 0
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = max_id + i + 1
                chunk_text = clean_text(chunk)
                verify_embedding_query = f"""
                SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_1024('voyage-multilingual-2', '{chunk_text}') as embedding
                """
                session.sql(verify_embedding_query).collect()
                # If embedding works, proceed with insertion
                insert_query = f"""
                INSERT INTO temp (chunk_id, chunk_text, text_vec)
                SELECT 
                    {chunk_id},
                    '{chunk_text}',
                    SNOWFLAKE.CORTEX.EMBED_TEXT_1024('voyage-multilingual-2', '{chunk_text}')
                """
                session.sql(insert_query).collect()
                successful_inserts += 1

            except Exception as chunk_error:
                st.warning(f"Error processing chunk {i + 1}: {str(chunk_error)}")
                continue

        # Verify data was inserted or not
        verification_query = "SELECT COUNT(*) as count FROM temp"
        count = session.sql(verification_query).collect()[0]['COUNT']
        if successful_inserts > 0:
            return True, f"Successfully processed {successful_inserts} chunks from PDF. Total records in temp table: {count}"
        else:
            return False, "Failed to insert any chunks into the database"
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"
def get_combined_context(question):
    try:
        # Get context from temp table
        temp_context_query = f"""
        SELECT chunk_text, VECTOR_COSINE_SIMILARITY(
            SNOWFLAKE.CORTEX.EMBED_TEXT_1024('voyage-multilingual-2', '{question.replace("'", "''")}'),
            text_vec
        ) AS similarity
        FROM temp
        ORDER BY similarity DESC
        LIMIT 7
        """
        temp_context = session.sql(temp_context_query).to_pandas()
        # Get context from medline table
        medline_context_query = f"""
        SELECT C1, C2, VECTOR_COSINE_SIMILARITY(
            SNOWFLAKE.CORTEX.EMBED_TEXT_1024('voyage-multilingual-2', '{question.replace("'", "''")}'),
            text_vec
        ) AS similarity
        FROM medline_ency_all
        ORDER BY similarity DESC
        LIMIT 5
        """
        medline_context = session.sql(medline_context_query).to_pandas()

        # Store contexts in session state
        st.session_state.context_data = {
            'temp': temp_context if not temp_context.empty else None,
            'medline': medline_context if not medline_context.empty else None
        }
        # Combine contexts
        combined_text = ""
        if not temp_context.empty:
            combined_text += "\nFrom uploaded documents:\n" + " ".join(temp_context['CHUNK_TEXT'].fillna(""))
        if not medline_context.empty:
            combined_text += "\nFrom medical database:\n" + " ".join(medline_context['C2'].fillna(""))

        return combined_text
    except Exception as e:
        st.error(f"Error in context retrieval: {str(e)}")
        return ""

# Streamlit setup
st.title("RAG 'n' ROLL | Disease Diagnosis Using RAG-LLM")

# Initialize session state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "context_data" not in st.session_state:
    st.session_state.context_data = None

# Sidebar content
st.sidebar.header("About the App")
st.sidebar.info(
    """
    This application uses RAG-LLM for medical diagnosis 
    - Medlical database (22,000 Rows of CSV File)
    - Your uploaded PDF documents
    - View Data 
    - Performance Analysis | Trulens
    """
)
# Modified task selection
# Initialize session state for tracking metrics
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = {
        'timestamp': [],
        'question': [],
        'response': [],
        'context': [],
        'response_time': [],
        'feedback_scores': []
    }

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Modified task selection
task = st.sidebar.radio(
    "Select Task",
    ["Ask Question", "View Data", "Upload PDF", "View Performance"]
)
if task == "Ask Question":
    st.subheader("Chat with the Assistant")

    # Display chat history
    for q, a in st.session_state.qa_history:
        with st.chat_message("user"):
            if "\n" in q:
                st.markdown(f"```\n{q}\n```")
            else:
                st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # Initialize the question key in session state if it doesn't exist
    if "question_key" not in st.session_state:
        st.session_state.question_key = 0

    question = st.text_area(
        "Your question:",
        key=f"question_input_{st.session_state.question_key}",
        placeholder="Type your question here...",
        height=100
    )

    if st.button("Submit"):
        if question:
            with st.spinner("Generating response..."):
                try:
                    start_time = datetime.now()
                    context = get_combined_context(question)
                    # Build conversation history
                    history_context = "\n".join([f"Q: {q} A: {a}" for q, a in
                                                 st.session_state.qa_history]) if st.session_state.qa_history else ""
                    if history_context:
                        history_context = f"Previous conversation:\n{history_context}\n\n"

                    full_prompt = f"""
                                        You are a friendly and knowledgeable Medical Assistant. Analyze the given context and question carefully to provide helpful, accurate responses.

                                        Context: {context}

                                        {history_context}

                                        Instructions for responding:
                                        - If the user asks to guess the disease, don't say you can't guess the disease, just ask for all the symptoms and based on the context try to guess. and also acknowledge this is may be wrong, but you should consult health specialist
                                        - For medical queries: Analyze symptoms, diagnose potential conditions, guess the disease if the user asks, explain causes, and suggest general treatments based on the provided context. Dont provide any links that user can refer
                                        - For document-specific questions: Reference and explain information directly from the uploaded PDF documents
                                        - If the User asks, like , i will give you pdf, you have to say , sure i will answer based on PDF
                                        - For general queries: Provide friendly, informative responses while maintaining medical professionalism, if user asks about you or what can you do, answer generally, im a medical assistant
                                        - Always be empathetic, clear, and supportive in your responses
                                        - If the context doesn't contain enough information to fully answer the question, acknowledge this and provide general guidance based on what is known

                                        Question: {question}
                                        Answer:
                                        """

                    # Get LLM response
                    rag_query = f"""
                                        SELECT snowflake.cortex.complete(
                                            'mistral-large',
                                            '{full_prompt.replace("'", "''")}'
                                        ) AS response
                                        """
                    response_data = session.sql(rag_query).to_pandas()
                    if not response_data.empty:
                        response = response_data.iloc[0, 0]
                        # response time caluclation
                        end_time = datetime.now()
                        response_time = (end_time - start_time).total_seconds()
                        # Feedback prompt with context relevance and groundedness scores, use trulesn fix before submission
                        feedback_prompt = f"""
                        You are a specialized evaluator for a Medical Assistant and PDF Query Response system. Your role is to assess how well the response addresses medical-related queries or questions based on the provided PDF context (if any). 
                        I will provide you with three pieces of information: the Question, the Context, and the Response.
                        
                        Based on these three, please give me two ratings between 1 and 5:
                        - Context Relevance - The first rating is for **Context Relevance**: How well does the response address the question, considering the provided context? if the provided context is not related to the question, give the lowest score eg - 1 
                        If the context is unrelated to the question (e.g., asking about medical advice but the context is unrelated), give a low score (e.g., 1 or 2).  
                        
                        - Groundedness -  The second rating is for **Groundedness**: How accurate and factually correct is the response, based on the given context? , 5 represents 100% correct, if you know the response is 100% correct or exactly present in Context then give 5 other wise , based on the context, give score! 
                        - If the question is not related to medical thing, but still the response is ok! its not misleading, then its ok to give average score,2 or 2.5 or 3
                         If the response makes claims that are unsupported, incorrect,  reduce the score accordingly.  
                        - If the context is absent or irrelevant to the question, score groundedness based on how accurately the response aligns with the expected domain knowledge (medical or PDF content).  
                        
                        **Instructions:**
                         Only evaluate **medical-related queries** or **PDF content queries**. If the question is unrelated to these domains, rate both **Context Relevance** and **Groundedness** as 1 or 2.
                        - For each aspect, rate on a scale of 1 to 5. You may use decimals (e.g., 3.5, 4.25) to indicate nuanced scoring, but remember that 5 represents 99% accuracy and appropriateness. 
                        - Provide only the two scores, separated by a comma (e.g., 2.5, 4). Do not add explanations or additional text.  
                        
                        
                        Here are the details for your analysis:
                        - **Question**: {question}
                        - **Context**: {context}
                        - **Response**: {response}
                        
                        Please return only the two ratings (e.g., 2, 4), nothing else.
                        """

                        feedback_query = f"""
                                            SELECT snowflake.cortex.complete(
                                                'mistral-large',
                                                '{feedback_prompt.replace("'", "''")}'
                                            ) AS score
                                            """
                        feedback_result = session.sql(feedback_query).to_pandas()
                        try:
                            feedback_score_raw = feedback_result.iloc[0]['SCORE'].strip()
                            relevance_score, groundedness_score = map(float, feedback_score_raw.split(','))
                        except (ValueError, IndexError):
                            relevance_score, groundedness_score = 0.0, 0.0  # Default scores if parsing fails
                            st.warning(
                                "Could not parse feedback scores. Defaulting to 0.0 for both relevance and groundedness.")

                        # Append to session state
                        st.session_state.qa_history.append((question, response))
                        st.session_state.metrics_history['timestamp'].append(datetime.now())
                        st.session_state.metrics_history['question'].append(question)
                        st.session_state.metrics_history['response'].append(response)
                        st.session_state.metrics_history['context'].append(context)
                        st.session_state.metrics_history['response_time'].append(response_time)
                        st.session_state.metrics_history['feedback_scores'].append({
                            "context_relevance": relevance_score,
                            "groundedness": groundedness_score
                        })
                    # After successful response
                    st.session_state.question_key += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please enter a question.")
# Task handling
elif task == "View Data":
    st.subheader("View Data")
    tab1, tab2 = st.tabs(["Context Data", "All Data"])
    with tab1:
        if st.session_state.context_data:
            st.write("### Relevant Context Data Used for Last Question")
            if st.session_state.context_data['medline'] is not None:
                st.write("#### From Medline Database:")
                st.dataframe(st.session_state.context_data['medline'])
            if st.session_state.context_data['temp'] is not None:
                st.write("#### From Uploaded Documents:")
                st.dataframe(st.session_state.context_data['temp'])
        else:
            st.info("Ask a question to see relevant context data.")
    with tab2:
        st.write("#### Medline Data:")
        medline_data = session.sql("SELECT C1, C2 FROM medline_ency_all LIMIT 10").to_pandas()
        st.dataframe(medline_data)
        st.write("#### Uploaded PDF Data:")
        temp_data = session.sql("SELECT chunk_id, chunk_text FROM temp").to_pandas()
        st.dataframe(temp_data)

elif task == "View Performance":
    st.title("RAG System Performance Analytics")
    if len(st.session_state.metrics_history['timestamp']) == 0:
        st.info("No performance data available yet. Start asking questions to generate metrics.")
    else:
        # Create DataFrame for visualization
        metrics_df = pd.DataFrame({
            'Timestamp': st.session_state.metrics_history['timestamp'],
            'Question': st.session_state.metrics_history['question'],
            'Response': st.session_state.metrics_history['response'],
            'Context': st.session_state.metrics_history['context'],
            'Response Time (s)': st.session_state.metrics_history['response_time'],
            'Context Relevance': [score['context_relevance'] for score in
                                  st.session_state.metrics_history['feedback_scores']],
            'Groundedness': [score['groundedness'] for score in st.session_state.metrics_history['feedback_scores']]
        })
        # Summary metrics - average
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_response_time = metrics_df['Response Time (s)'].mean()
            st.metric("Average Response Time", f"{avg_response_time:.2f}s")
        with col2:
            avg_context_relevance = metrics_df['Context Relevance'].mean()
            st.metric("Average Context Relevance", f"{avg_context_relevance:.2f} / 5")
        with col3:
            avg_groundedness = metrics_df['Groundedness'].mean()
            st.metric("Average Groundedness", f"{avg_groundedness:.2f} / 5")
        # Plot 1: Response Time to Time
        fig1 = px.line(
            metrics_df,
            x='Timestamp',
            y='Response Time (s)',
            title='Response Time Over Time',
            labels={'Response Time (s)': 'Response Time (seconds)'}
        )
        st.plotly_chart(fig1)
        # Plot 2: Context Relevance over Time
        fig2 = px.line(
            metrics_df,
            x='Timestamp',
            y='Context Relevance',
            title='Context Relevance Over Time',
            labels={'Context Relevance': 'Context Relevance Score'}
        )
        st.plotly_chart(fig2)
        # Plot 3: Groundedness over Time
        fig3 = px.line(
            metrics_df,
            x='Timestamp',
            y='Groundedness',
            title='Groundedness Over Time',
            labels={'Groundedness': 'Groundedness Score'}
        )
        st.plotly_chart(fig3)
        # Detailed metrics table
        st.subheader("Detailed Query History")
        metrics_df['Timestamp'] = metrics_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(metrics_df)

elif task == "Upload PDF":
    st.subheader("Upload PDF Document")
    # Add table verification
    table_status = verify_temp_table()
    st.info(f"Table Status: {table_status}")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            success, message = process_pdf(uploaded_file)
            if success:
                st.success(message)
                # Show current data in temp table
                temp_data = session.sql(
                    "SELECT chunk_id, chunk_text FROM temp ORDER BY chunk_id DESC LIMIT 5").to_pandas()
                st.write("Recently added chunks:")
                st.dataframe(temp_data)
            else:
                st.error(message)
