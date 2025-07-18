import streamlit as st
import requests
import os
import mysql.connector
import bcrypt
from dotenv import load_dotenv

load_dotenv()
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]

HOST_VAL = os.getenv("HOST_VAL")
PORT_VAL = os.getenv("PORT_VAL")
USER_VAL = os.getenv("USER_VAL")
PASSWORD_VAL = os.getenv("PASSWORD_VAL")
DATABASE_VAL = os.getenv("DATABASE_VAL")

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode() 

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def connect_db():
    return mysql.connector.connect(
        host = HOST_VAL,
        port = int(PORT_VAL),
        user = USER_VAL,
        password = PASSWORD_VAL,
        database = DATABASE_VAL
    )

def ask_together(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"[TOGETHER API ERROR] {response.status_code} - {response.text}"

def is_db_question(prompt):
    db_keywords = [
    "list", "show", "find", "get", "fetch", "display", "view", "retrieve", "mileage", "engine", "price", "brand", "model", "year", "manufacture", "color", "quantity", "available", "stock", "petrol", "diesel", "ev", "electric", "fuel", "engine type", "battery", "hybrid", "above", "below", "under", "over", "greater", "less", "between", "top", "lowest", "highest", "toyota", "honda", "tata", "hyundai", "mahindra", "kia", "carens", "creta", "nexon", "xuv", "city", "car", "cars", "vehicle", "vehicles", "models"]
    return any(word in prompt.lower() for word in db_keywords)

def execute_db_query(query):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        if not rows:
            return "No results found."
        result = ""
        for row in rows:
            row_str = ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
            result += row_str + "\n\n"
        return result.strip()
    
    except Exception as e:
        return f"DB Error:\n{str(e)}\n\nQuery:\n{query}"


def ask_combined(prompt):
    if is_db_question(prompt):
        sql_prompt =  f"""
        You are an expert SQL assistant.
        Given a natural language question, convert it into a valid MySQL SELECT query using the following two tables:
        Table: car as c
        - car_id (int, primary key)
        - brand (varchar)
        - model (varchar)
        - manufacture_year (int)
        - color (varchar)
        - price (int)
        - quantity_available (int)
        Table: spec as s
        - spec_id (int, primary key)
        - car_id (foreign key to car.car_id)
        - engine_type (varchar)
        - mileage (int)
        Instructions:
        - Use only the column names listed above exactly as written.
        - If the user misspells a column name (e.g., "milege"), intelligently infer and map it to the correct column name (e.g., "mileage").
        - If the user misspells a data value (e.g., "Toyata" instead of "Toyota", or "disel" instead of "diesel"), intelligently infer and correct it using common known values.

        When generating SQL queries based on user questions:

        1. If the user asks:
        - "how many cars"
        - "total quantity"
        - "number of cars"
        - "cars available"
        → Use:
        SELECT SUM(c.quantity_available) FROM car c JOIN spec s ON c.car_id = s.car_id WHERE ...

        2. If the user asks:
        -"how much will it cost"
        - "total cost"
        - "value of cars"
        - "price to sell all cars"
        → Use:
        SELECT SUM(c.price * c.quantity_available) FROM car c JOIN spec s ON c.car_id = s.car_id WHERE ...

        3. If the user asks:
        - "price of [car model]"
        - "cost of one [car model]"
        - "how much is [car model]"
        → Use:
        SELECT c.price FROM car c WHERE c.model = '...';

        → However, if the user includes:
        - "all [model] cars"
        - "total price of [model]"
        - "cost of all [model]"
        → Then use:
        SELECT SUM(c.price * c.quantity_available) FROM car c JOIN spec s ON c.car_id = s.car_id WHERE c.model = '...';

        4. Use JOIN between `car` and `spec` on `car_id` when filtering by `engine_type`, `mileage`, etc.

        # Examples:
        Q: How many red Kia cars are available?
        A: SELECT SUM(c.quantity_available) FROM car c JOIN spec s ON c.car_id = s.car_id WHERE c.brand = 'Kia' AND c.color = 'Red';

        Q: How much will it cost to sell all Tata petrol cars?
        A: SELECT SUM(c.price * c.quantity_available) FROM car c JOIN spec s ON c.car_id = s.car_id WHERE c.brand = 'Tata' AND s.engine_type = 'Petrol';

        Q: What is the price of EV6?
        A: SELECT c.price FROM car c WHERE c.model = 'EV6';

        Q: What is the price of all Fortuner cars?
        A: SELECT SUM(c.price * c.quantity_available) FROM car c JOIN spec s ON c.car_id = s.car_id WHERE c.model = 'Fortuner';

        - Use contextual understanding to resolve close matches in brand, engine_type, color, model, etc.
        - Assume the user is asking about car inventory and specifications.
        - Add appropriate WHERE clauses based on the user's question.
        - Use JOINs when needed to include fields from both `car` and `spec` tables.
        - Always use correct column names from the given schema.
        - For questions like "highest", "lowest", "top", or similar, return all relevant columns **except car_id**.
        - Do NOT include 'car_id' in the SELECT clause unless the user explicitly asks for it using phrases like "show car_id", "include car_id", or "give me car_id".
        - Do not include backticks or markdown formatting (no ```sql or ```).
        - Return ONLY the final valid MySQL SELECT query.
        - Do not explain, describe, or annotate the query.

        Now convert the following question to SQL:
        "{prompt}"
        """
        sql_query = ask_together(sql_prompt)
        if sql_query.startswith("[TOGETHER API ERROR]"):
            return sql_query
        return execute_db_query(sql_query)
    else:
        return ask_together(prompt)

def admin_dashboard():
    st.title("Admin Dashboard")
    tab_1,tab_2,tab_3 = st.tabs(["User","Logs","Car Inventory"])

    with tab_1:
        st.subheader("User Registered")
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT email, role FROM users WHERE email != %s", (st.session_state.user_email,))
        users = cursor.fetchall()
        cursor.close()
        conn.close()
        if not users:
            st.info("No other users found.")
        else:
            for i,(email,role) in enumerate(users):
                col_1,col_2,col_3 = st.columns([4,3,5])
                col_1.write(email)
                col_2.write(role)

                with col_3.expander("Action"):
                    new_role = st.radio(
                        f"Change role for {email}",
                        ["user", "admin"],
                        index=0 if role == "user" else 1,
                        key=f"role_{i}",
                        horizontal=True
                    )
                    if new_role != role:
                        if st.button("Update Role",key=f"update_btn_{i}"):
                            conn = connect_db()
                            cur = conn.cursor()
                            cur.execute("UPDATE users SET role = %s WHERE email = %s", (new_role, email))
                            conn.commit()
                            cur.close()
                            conn.close()
                            st.success(f"Role updated to {new_role} for {email}")
                            st.rerun()

                    if st.button(f"Delete {email}", key=f"delete_btn_{i}"):
                        conn = connect_db()
                        cur = conn.cursor()
                        cur.execute("DELETE FROM users WHERE email = %s", (email,))
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.warning(f"{email} has been deleted.")
                        st.rerun()

    with tab_2:
        st.subheader("User Login Time")
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT email, login_time FROM login_logs ORDER BY login_time DESC")
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        if data:
            st.dataframe(
            [{"Email": row[0], "Login Time": row[1]} for row in data],
            hide_index=True
            )
        else:
            st.info("No Login Recorded")

    with tab_3:
        st.subheader("Car Inventory and Specifications")
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT car.car_id, brand, model, manufacture_year, color, price, quantity_available, engine_type, mileage
            FROM car
            JOIN spec 
            ON car.car_id = spec.car_id
        """)
        rows = cursor.fetchall()
        columns = ["Car ID", "Brand", "Model", "Year", "Color", "Price", "Stock", "Engine Type", "Mileage"]
        cursor.close()
        conn.close()
        if rows:
            st.dataframe([dict(zip(columns, row)) for row in rows])
        else:
            st.info("No cars found")


st.set_page_config(page_title="Super Wheels", page_icon="🚙", layout="centered")

if "free_used" not in st.session_state:
    st.session_state.free_used = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None

def login():
    st.markdown("""
        <style>
        .stApp {
            background-color: #3366ff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Login / Sign up")

    if "last_action" not in st.session_state:
        st.session_state.last_action = "Login"

    def switch_action():
        st.session_state.email = ""
        st.session_state.password = ""
        st.session_state.last_action = st.session_state.action

    action = st.radio("Choose an Option", ["Login", "Sign up"], key="action", on_change=switch_action)
    email = st.text_input("Email ID", placeholder="Enter Email", key="email")
    password = st.text_input("Password", type="password", placeholder="Enter Password", key="password")

    if action=="Sign up":
        if st.button("Create Account"):
            conn = connect_db()
            cursor = conn.cursor()
            try:
                hashed_pw = hash_password(password)
                cursor.execute("INSERT INTO users (email, password, role) VALUES (%s, %s, 'user')", (email, hashed_pw))
                conn.commit()
                st.success("Account Created Successfully")
            except mysql.connector.errors.IntegrityError:
                st.error("Email Already Exists")
            cursor.close()
            conn.close()

    elif action == "Login":
        if st.button("Login"):
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("SELECT password, role FROM users WHERE email = %s",(email,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row and check_password(password,row[0]):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.user_role = row[1]

                log_conn = connect_db()
                log_cursor = log_conn.cursor()
                log_cursor.execute("INSERT INTO login_logs (email) VALUES (%s)", (email,))
                log_conn.commit()
                log_cursor.close()
                log_conn.close()

                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials.")

def chatbot():
    st.title("Super Wheels Chatbot")
    question = st.chat_input("Ask Something...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        answer = ask_combined(question)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if not st.session_state.free_used and not st.session_state.logged_in:
    st.title("Super Wheels Chatbot")
    st.info("You get 1 free question without login!")
    question = st.chat_input("Ask Something...")
    if question:
        answer = ask_combined(question)
        st.chat_message("user").markdown(question)
        st.chat_message("assistant").markdown(answer)
        st.session_state.free_used = True
        st.success("Free question used. Please login to continue.")
        if st.button("Login Now"):
            st.session_state.force_login = True
            st.rerun()

else:
    if not st.session_state.logged_in:
        login()
    else:
        if st.session_state.user_role == "admin":
            st.sidebar.title("Admin Panel")
            view = st.sidebar.radio("Navigate", ["Dashboard", "Chatbot"])

            if view == "Dashboard":
                admin_dashboard()
            else:
                chatbot()
        else:
            chatbot()
