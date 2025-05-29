try:
    import sqlite3
except ImportError:
    try:
        import sqlite4 as sqlite3
        print("Using sqlite4 as sqlite3")
    except ImportError:
        raise ImportError("Neither sqlite3 nor sqlite4 is available")

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import gradio as gr
from typing import List, Dict, Any, Tuple
import json
import os
from datetime import datetime

class TextToSQLApp:
    def __init__(self, model_name: str = "deepseek-coder:6.7b"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)
        self.db_connection = None
        self.schema_info = {}
        self.conversation_history = []
        
        # Enhanced prompt template with schema awareness
        self.template = """You are an expert SQL assistant that converts natural language questions into SQL queries.

Database Schema Information:
{schema_info}

Guidelines:
1. Always use correct SQL syntax for SQLite
2. Use table and column names exactly as they appear in the schema
3. Include DISTINCT when needed to avoid duplicates
4. Use LIMIT for result size control
5. Use proper JOINs when querying multiple tables
6. Handle date/time columns appropriately
7. If the question is unclear, provide the best possible interpretation
8. Only return the SQL query, no explanations unless specifically asked

Previous Context: {context}

Question: {question}

SQL Query:"""

        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm

    def connect_database(self, db_path: str) -> str:
        """Connect to SQLite database and extract schema information"""
        try:
            # Try different connection methods
            if hasattr(sqlite3, 'connect'):
                self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
            else:
                # Alternative connection method for sqlite4
                self.db_connection = sqlite3.Connection(db_path)
            
            self.schema_info = self._extract_schema()
            return f"✅ Connected to database: {db_path}\n\nSchema loaded with {len(self.schema_info)} tables"
        except Exception as e:
            return f"❌ Error connecting to database: {str(e)}\nSQLite module info: {sqlite3}"

    def _extract_schema(self) -> Dict[str, Any]:
        """Extract detailed schema information from the database"""
        schema = {}
        cursor = self.db_connection.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            schema[table_name] = {
                'columns': [],
                'sample_data': []
            }
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            for col in columns:
                schema[table_name]['columns'].append({
                    'name': col[1],
                    'type': col[2],
                    'not_null': col[3],
                    'primary_key': col[5]
                })
            
            # Get sample data (first 3 rows)
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                
                for row in sample_rows:
                    schema[table_name]['sample_data'].append(dict(zip(column_names, row)))
            except:
                pass  # Skip if there's an error getting sample data
        
        return schema

    def format_schema_info(self) -> str:
        """Format schema information for the prompt"""
        if not self.schema_info:
            return "No database connected."
        
        formatted = "Available Tables:\n"
        for table_name, info in self.schema_info.items():
            formatted += f"\nTable: {table_name}\n"
            formatted += "Columns:\n"
            for col in info['columns']:
                pk = " (PRIMARY KEY)" if col['primary_key'] else ""
                nn = " NOT NULL" if col['not_null'] else ""
                formatted += f"  - {col['name']}: {col['type']}{pk}{nn}\n"
            
            if info['sample_data']:
                formatted += "Sample Data:\n"
                for i, row in enumerate(info['sample_data'][:2]):  # Show max 2 sample rows
                    formatted += f"  Row {i+1}: {row}\n"
        
        return formatted

    def execute_query(self, sql_query: str) -> Tuple[str, pd.DataFrame]:
        """Execute SQL query and return results"""
        if not self.db_connection:
            return "❌ No database connected", pd.DataFrame()
        
        try:
            # Clean the SQL query
            sql_query = sql_query.strip()
            if sql_query.startswith('```sql'):
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Execute query
            df = pd.read_sql_query(sql_query, self.db_connection)
            
            if df.empty:
                return "✅ Query executed successfully but returned no results", df
            else:
                return f"✅ Query executed successfully. Found {len(df)} rows", df
        
        except Exception as e:
            return f"❌ SQL Error: {str(e)}", pd.DataFrame()

    def generate_sql(self, question: str, include_context: bool = True) -> str:
        """Generate SQL query from natural language question"""
        try:
            # Prepare context from recent conversation
            context = ""
            if include_context and self.conversation_history:
                recent_context = self.conversation_history[-3:]  # Last 3 exchanges
                context = "\n".join([f"Q: {item['question']}\nSQL: {item['sql']}" 
                                   for item in recent_context])
            
            # Generate SQL
            result = self.chain.invoke({
                "question": question,
                "schema_info": self.format_schema_info(),
                "context": context
            })
            
            # Clean up the result
            sql_query = result.strip()
            if sql_query.startswith('```sql'):
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "sql": sql_query,
                "timestamp": datetime.now().isoformat()
            })
            
            return sql_query
        
        except Exception as e:
            return f"Error generating SQL: {str(e)}"

    def create_sample_database(self) -> str:
        """Create a sample database for testing"""
        try:
            # Create sample database
            sample_db_path = "sample_ecommerce.db"
            
            # Try different connection methods
            if hasattr(sqlite3, 'connect'):
                conn = sqlite3.connect(sample_db_path)
            else:
                conn = sqlite3.Connection(sample_db_path)
            
            cursor = conn.cursor()
            
            # Create tables with IF NOT EXISTS
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id INTEGER PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    registration_date DATE,
                    city TEXT,
                    country TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id INTEGER PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    category TEXT,
                    price DECIMAL(10,2),
                    stock_quantity INTEGER,
                    supplier TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    order_date DATE,
                    total_amount DECIMAL(10,2),
                    status TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_items (
                    item_id INTEGER PRIMARY KEY,
                    order_id INTEGER,
                    product_id INTEGER,
                    quantity INTEGER,
                    unit_price DECIMAL(10,2),
                    FOREIGN KEY (order_id) REFERENCES orders (order_id),
                    FOREIGN KEY (product_id) REFERENCES products (product_id)
                )
            """)
            
            # Insert sample data
            customers_data = [
                (1, 'John', 'Doe', 'john.doe@email.com', '2023-01-15', 'New York', 'USA'),
                (2, 'Jane', 'Smith', 'jane.smith@email.com', '2023-02-20', 'London', 'UK'),
                (3, 'Bob', 'Johnson', 'bob.johnson@email.com', '2023-03-10', 'Toronto', 'Canada'),
                (4, 'Alice', 'Brown', 'alice.brown@email.com', '2023-04-05', 'Sydney', 'Australia'),
                (5, 'Charlie', 'Wilson', 'charlie.wilson@email.com', '2023-05-12', 'Berlin', 'Germany')
            ]
            
            products_data = [
                (1, 'iPhone 14', 'Electronics', 999.99, 50, 'Apple Inc'),
                (2, 'Samsung Galaxy S23', 'Electronics', 849.99, 30, 'Samsung'),
                (3, 'Nike Air Max', 'Footwear', 129.99, 100, 'Nike'),
                (4, 'Adidas Ultraboost', 'Footwear', 189.99, 75, 'Adidas'),
                (5, 'MacBook Pro', 'Electronics', 1999.99, 25, 'Apple Inc'),
                (6, 'Coffee Maker', 'Appliances', 79.99, 40, 'Breville')
            ]
            
            orders_data = [
                (1, 1, '2023-06-01', 999.99, 'Completed'),
                (2, 2, '2023-06-02', 259.98, 'Completed'),
                (3, 3, '2023-06-03', 1999.99, 'Processing'),
                (4, 1, '2023-06-04', 209.98, 'Shipped'),
                (5, 4, '2023-06-05', 849.99, 'Completed')
            ]
            
            order_items_data = [
                (1, 1, 1, 1, 999.99),  # iPhone 14
                (2, 2, 3, 2, 129.99),  # 2x Nike Air Max
                (3, 3, 5, 1, 1999.99), # MacBook Pro
                (4, 4, 4, 1, 189.99),  # Adidas Ultraboost
                (5, 4, 6, 1, 79.99),   # Coffee Maker
                (6, 5, 2, 1, 849.99)   # Samsung Galaxy S23
            ]
            
            # Use INSERT OR REPLACE to handle existing data
            cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?,?)", customers_data)
            cursor.executemany("INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?)", products_data)
            cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?)", orders_data)
            cursor.executemany("INSERT OR REPLACE INTO order_items VALUES (?,?,?,?,?)", order_items_data)
            
            conn.commit()
            conn.close()
            
            # Connect to the created database
            return self.connect_database(sample_db_path)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"❌ Error creating sample database: {str(e)}\n\nDetailed error:\n{error_details}\n\nSQLite module: {sqlite3}\nSQLite version: {getattr(sqlite3, 'sqlite_version', 'Unknown')}"

# Initialize the app
app = TextToSQLApp()

def chat_interface(message, history):
    """Main chat interface function"""
    if not app.db_connection:
        return "Please connect to a database first using the Database Connection tab."
    
    # Generate SQL
    sql_query = app.generate_sql(message)
    
    # Execute SQL and get results
    result_message, df = app.execute_query(sql_query)
    
    # Format response
    response = f"**Generated SQL:**\n```sql\n{sql_query}\n```\n\n"
    response += f"**Execution Result:** {result_message}\n\n"
    
    if not df.empty:
        # Show first 10 rows of results
        display_df = df.head(10)
        response += f"**Results Preview (showing {len(display_df)} of {len(df)} rows):**\n"
        response += display_df.to_string(index=False)
        
        if len(df) > 10:
            response += f"\n\n... and {len(df) - 10} more rows"
    
    return response

def connect_db(db_file):
    """Handle database connection"""
    if db_file is None:
        return "Please upload a database file."
    
    try:
        return app.connect_database(db_file.name)
    except Exception as e:
        return f"Error: {str(e)}"

def create_sample_db():
    """Create sample database"""
    return app.create_sample_database()

def execute_custom_sql(sql_query):
    """Execute custom SQL query"""
    if not sql_query.strip():
        return "Please enter a SQL query.", ""
    
    result_message, df = app.execute_query(sql_query)
    
    if df.empty:
        return result_message, ""
    else:
        return result_message, df

# Create Gradio interface
with gr.Blocks(title="Advanced Text-to-SQL Application", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Advanced Text-to-SQL Application")
    gr.Markdown("Convert natural language questions into SQL queries and execute them against your database.")
    
    with gr.Tabs():
        # Main Chat Tab
        with gr.TabItem("💬 Chat Interface"):
            gr.Markdown("Ask questions about your data in natural language!")
            
            chatbot = gr.ChatInterface(
                fn=chat_interface,
                title="Text-to-SQL Chat",
                description="Type your question about the data...",
                examples=[
                    "Show me all customers from USA",
                    "What are the top 5 best selling products?",
                    "Find customers who have spent more than $500",
                    "Show me orders placed in the last month",
                    "Which products are low in stock?"
                ]
            )
        
        # Database Connection Tab
        with gr.TabItem("🗄️ Database Connection"):
            gr.Markdown("### Connect to your SQLite database or create a sample one")
            
            with gr.Row():
                with gr.Column():
                    db_file = gr.File(
                        label="Upload SQLite Database (.db file)",
                        file_types=[".db", ".sqlite", ".sqlite3"]
                    )
                    connect_btn = gr.Button("Connect to Database", variant="primary")
                
                with gr.Column():
                    sample_btn = gr.Button("Create Sample E-commerce Database", variant="secondary")
            
            connection_status = gr.Textbox(
                label="Connection Status",
                placeholder="No database connected",
                interactive=False,
                lines=10
            )
            
            connect_btn.click(connect_db, inputs=[db_file], outputs=[connection_status])
            sample_btn.click(create_sample_db, outputs=[connection_status])
        
        # SQL Executor Tab
        with gr.TabItem("⚡ SQL Executor"):
            gr.Markdown("### Execute custom SQL queries directly")
            
            sql_input = gr.Textbox(
                label="SQL Query",
                placeholder="Enter your SQL query here...",
                lines=5
            )
            
            execute_btn = gr.Button("Execute Query", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    execution_result = gr.Textbox(
                        label="Execution Status",
                        interactive=False,
                        lines=3
                    )
                with gr.Column():
                    query_results = gr.Dataframe(
                        label="Query Results",
                        interactive=False
                    )
            
            execute_btn.click(
                execute_custom_sql,
                inputs=[sql_input],
                outputs=[execution_result, query_results]
            )
        
        # Help Tab
        with gr.TabItem("❓ Help & Examples"):
            gr.Markdown("""
            ### How to Use This Application
            
            1. **Connect Database**: Upload your SQLite database file or create a sample database
            2. **Ask Questions**: Use natural language to query your data
            3. **Review Results**: See the generated SQL and execution results
            
            ### Example Questions You Can Ask:
            
            **Basic Queries:**
            - "Show me all customers"
            - "List all products in the Electronics category"
            - "How many orders do we have?"
            
            **Advanced Queries:**
            - "Find the top 10 customers by total spending"
            - "Show me monthly sales trends"
            - "Which products have never been ordered?"
            - "Calculate average order value by country"
            
            **Aggregations:**
            - "What's the total revenue this year?"
            - "Show me the best selling product categories"
            - "Count orders by status"
            
            ### Tips for Better Results:
            - Be specific about what data you want to see
            - Mention if you want results sorted or limited
            - Use table and column names if you know them
            - Ask follow-up questions to refine your queries
            """)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
