import sqlite3
import pandas as pd
import random

# Connect to (or create) SQLite database
conn = sqlite3.connect("company.db")
cursor = conn.cursor()

# Drop tables if they exist
cursor.execute("DROP TABLE IF EXISTS employees")
cursor.execute("DROP TABLE IF EXISTS departments")

# Create tables
cursor.execute("""
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    salary INTEGER,
    FOREIGN KEY(department_id) REFERENCES departments(id)
)
""")

# Sample department names
department_names = ["Engineering", "HR", "Sales", "Marketing", "Finance"]

# Insert departments
for i, name in enumerate(department_names, start=1):
    cursor.execute("INSERT INTO departments (id, name) VALUES (?, ?)", (i, name))

# Generate fake employees
names = ["Alice", "Bob", "Charlie", "Diana", "Edward", "Fay", "George", "Hannah", "Ian", "Jane"]
for i in range(1, 21):
    name = random.choice(names) + f"_{i}"
    dept_id = random.randint(1, len(department_names))
    salary = random.randint(30000, 120000)
    cursor.execute(
        "INSERT INTO employees (id, name, department_id, salary) VALUES (?, ?, ?, ?)",
        (i, name, dept_id, salary)
    )

# Commit and close
conn.commit()
conn.close()

print("✅ Synthetic database 'company.db' created!")
