# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install streamlit pandas matplotlib langchain 
# langchain-community langchain-groq python-dotenv



import streamlit as st
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI SQL Data Analyst")
st.title("📊 AI SQL Data Analyst Agent")

# ------------------ LOAD API ------------------ #
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ------------------ FILE UPLOAD ------------------ #
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ------------------ CREATE DATABASE ------------------ #
    conn = sqlite3.connect("data.db")
    df.to_sql("data_table", conn, if_exists="replace", index=False)

    st.success("✅ Data stored in SQLite database!")

    # ------------------ USER QUERY ------------------ #
    question = st.text_input("💬 Ask a question about your data")

    if question:

        # ------------------ PROMPT ------------------ #
        prompt = f"""
You are a SQL expert.

Table name: data_table
Columns: {list(df.columns)}

Write ONLY SQL query.

Rules:
- No explanation
- No markdown
- Only SQL

Question: {question}
"""

        # ------------------ LLM RESPONSE ------------------ #
        response = llm.invoke(prompt)
        query = response.content.strip()

        st.subheader("🧠 Generated SQL Query")
        st.code(query, language="sql")

        # ------------------ EXECUTE QUERY ------------------ #
        try:
            result_df = pd.read_sql_query(query, conn)

            st.subheader("📊 Result")
            st.dataframe(result_df)

            # ------------------ VISUALIZATION ------------------ #
            if not result_df.empty and len(result_df.columns) >= 2:

                st.subheader("📈 Visualization")

                x = result_df.iloc[:, 0]
                y = result_df.iloc[:, 1]

                fig, ax = plt.subplots()
                ax.bar(x, y)

                plt.xticks(rotation=45)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ SQL Error: {e}")