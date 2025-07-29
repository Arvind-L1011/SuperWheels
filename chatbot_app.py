import streamlit as st
import requests
import os
import mysql.connector
import bcrypt
import io
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
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

    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {"role": "user", "content": prompt}

    history = [msg for msg in st.session_state.chat_history if msg.get("content_type", "text") == "text"]
    messages = [system_msg] + [{"role": msg["role"], "content": msg["content"]} for msg in history] + [user_msg]

    def count_tokens(text):
        return len(text) // 4

    token_budget = 7000
    total_tokens = sum(count_tokens(msg["content"]) for msg in messages)

    while total_tokens > token_budget and len(messages) > 2:
        messages.pop(1)
        total_tokens = sum(count_tokens(msg["content"]) for msg in messages)

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

def user_requested_graph(prompt):
    graph_keywords = ["graph", "bar graph", "bar chart", "plot", "show graph", "chart"]
    return any(word in prompt.lower() for word in graph_keywords)

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

        remove_cols = {"car_id", "spec_id"}
        remove_indexes = [i for i, col in enumerate(columns) if col in remove_cols]

        for index in sorted(remove_indexes, reverse=True):
            columns.pop(index)
            rows = [tuple(val for i, val in enumerate(row) if i != index) for row in rows]

        result = ""
        for row in rows:
            row_str = ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
            result += row_str + "\n\n"
        return {"columns": columns, "rows": rows}

    except Exception as e:
        return {"error": "There was a problem executing your request. Please try again later."}

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
        - Do NOT include `car_id` or `spec_id` in the SELECT clause
        - Do not include backticks or markdown formatting (no ```sql or ```).
        - Only return valid SQL statements. Do not include any explanation or prefix.
        - Return ONLY the final valid MySQL SELECT query.
        - Do not explain, describe, or annotate the query. 

        Now convert the following question to SQL:
        "{prompt}"
        """
        sql_query = ask_together(sql_prompt)
        if sql_query.startswith("[TOGETHER API ERROR]"):
            return {"error": sql_query}
        return execute_db_query(sql_query)
    else:
        return ask_together(prompt)

def admin_dashboard():
    st.title("🧑‍💼 Admin Dashboard")
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
    st.markdown(
    """
    <style>
    section.main > div {
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0rem !important;
        border: none !important;
    }

    div[data-testid="stForm"], .block-container, .stContainer {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 5 !important;
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.title("Login / Sign up")

    if "last_action" not in st.session_state:
        st.session_state.last_action = "Login"

    def switch_action():
        st.session_state.email = ""
        st.session_state.password = ""
        st.session_state.last_action = st.session_state.action

    action = st.radio("Choose an Option", ["Login", "Sign up"], key="action", on_change=switch_action, horizontal=True)
    
    if action=="Sign up":
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input("Email ID", placeholder="Enter Email", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="Enter Password", key="signup_password")

            submitted = st.form_submit_button("Create Account")

            if submitted:
                with st.spinner("Creating a new account..."):
                    if not email or not password:
                        st.warning("Please enter the credentials.")
                    else:
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
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email ID", placeholder="Enter Email", key="login_email")
            password = st.text_input("Password", type="password", placeholder="Enter Password", key="login_password")
            
            submitted = st.form_submit_button("Login") 
            
            if submitted:
                with st.spinner("Logging you in..."):
                    if not email or not password:
                        st.warning("Please enter the credentials.")
                    else:
                        conn = connect_db()
                        cursor = conn.cursor()
                        cursor.execute("SELECT password, role FROM users WHERE email = %s", (email,))
                        row = cursor.fetchone()
                        cursor.close()
                        conn.close()

                        if row and check_password(password, row[0]):
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

def create_pdf_response(user, assistant, fig=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"<b>User:</b> {user}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    if isinstance(assistant, dict) and "columns" in assistant and "rows" in assistant:
        elements.append(Paragraph("<b>Assistant Response (Table):</b>", styles["Normal"]))
        elements.append(Spacer(1, 12))

        data = [assistant["columns"]] + list(assistant["rows"])
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.black),
        ]))
        elements.append(table)

        if fig:
            from reportlab.platypus import Image
            import tempfile

            elements.append(Spacer(1, 24))
            elements.append(Paragraph("<b>Graph:</b>", styles["Normal"]))
            elements.append(Spacer(1, 12))

            if hasattr(fig, "savefig"):
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmpfile.name, format="png", bbox_inches="tight")
                tmpfile.close()
                elements.append(Image(tmpfile.name, width=6 * inch, height=4 * inch))
            elif isinstance(fig, str):
                elements.append(Image(fig, width=6 * inch, height=4 * inch))

    else:
        elements.append(Paragraph(f"<b>Assistant:</b> {assistant}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_bar_graph(columns, rows):
    try:
        if not rows or len(columns) <2:
            return None
        x_vals = [str(row[0]) for row in rows]
        y_vals = [float(row[1])for row in rows]
        fig, ax = plt.subplots()
        ax.bar(x_vals,y_vals, color='blue')
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_title(f"{columns[1]} vs {columns[0]}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
            st.error(f"Error generating graph: {e}")
            return None     

def chatbot():
    st.title("🤖 Chatbot")
    question = st.chat_input("Fuel me with your questions...")
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            if msg.get("content_type") == "table":
                st.dataframe([dict(zip(msg["columns"], row)) for row in msg["rows"]])

                fig_path = None
                if "graph" in msg and msg["graph"]:
                    try:
                        import tempfile
                        from reportlab.platypus import Image
                        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        tmpfile.write(msg["graph"])
                        tmpfile.flush()
                        tmpfile.close()
                        fig_path = tmpfile.name
                        buf = io.BytesIO(msg["graph"])
                        img = plt.imread(buf, format='png')
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.imshow(img)
                        ax.axis("off")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Failed to reload graph image: {e}")

                for j in range(idx - 1, -1, -1):
                    if st.session_state.chat_history[j]["role"] == "user":
                        user_msg = st.session_state.chat_history[j]["content"]
                        break
                else:
                    user_msg = "Unknown"

                pdf_file = create_pdf_response(user_msg, {
                    "columns": msg["columns"],
                    "rows": msg["rows"]
                }, fig_path)

                st.download_button(
                    label="📄 Download",
                    data=pdf_file,
                    file_name="chat_table_response.pdf",
                    mime="application/pdf",
                    key=f"download_table_{idx}"
                )
            elif msg["role"] == "assistant":
                st.markdown(msg["content"])
                
                for j in range(idx - 1, -1, -1):
                    if st.session_state.chat_history[j]["role"] == "user":
                        user_msg = st.session_state.chat_history[j]["content"]
                        break
                else:
                    user_msg = "Unknown"

                pdf_file = create_pdf_response(user_msg, msg["content"])
                st.download_button(
                    label="📄 Download",
                    data=pdf_file,
                    file_name="chat_response.pdf",
                    mime="application/pdf",
                    key=f"download_text_{idx}"
                )

            else:
                st.markdown(msg["content"])
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Revving the engines...🚗💨"):
                answer = ask_combined(question)

            if isinstance(answer, dict) and "columns" in answer and "rows" in answer:
                st.dataframe([dict(zip(answer["columns"], row)) for row in answer["rows"]])

                fig = None
                fig_bytes = None
                if user_requested_graph(question):
                    fig = generate_bar_graph(answer["columns"], answer["rows"])
                    if fig:
                        st.pyplot(fig)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        fig_bytes = buf.read()
                    else:
                        st.error("Unable to generate graph")

                pdf_file = create_pdf_response(question, answer, fig)
                st.download_button(
                    label="📄 Download",
                    data=pdf_file,
                    file_name="query_response.pdf",
                    mime="application/pdf",
                    key=f"download_table_{len(st.session_state.chat_history)}"
                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content_type": "table",
                    "columns": answer["columns"],
                    "rows": answer["rows"],
                    "graph": fig_bytes
                })
            elif isinstance(answer, dict) and "error" in answer:
                st.error(answer["error"])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content_type": "text",
                    "content": answer["error"]
                })
            else:
                st.markdown(answer)
                pdf_file = create_pdf_response(question, answer)
                st.download_button(
                    label="📄 Download",
                    data=pdf_file,
                    file_name="chat_response.pdf",
                    mime="application/pdf",
                    key=f"download_button_{len(st.session_state.chat_history)}"
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content_type": "text",
                    "content": answer
                })

if not st.session_state.free_used and not st.session_state.logged_in:
    logo_img_url = "https://cdn-icons-png.flaticon.com/512/465/465077.png"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="{logo_img_url}" alt="icon" style="width:40px; height:40px; margin-right:10px;">
            <h1 style="margin: 0; font-size: 45px;">Super Wheels</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("🤖 Chatbot")
    st.info("You get 1 free question without login!")
    question = st.chat_input("Fuel me with your questions...")
    if question:
        st.chat_message("user").markdown(question)
        with st.spinner("Revving the engines...🚗💨"):
            answer = ask_combined(question)
        with st.chat_message("assistant"):
            if isinstance(answer, dict) and "columns" in answer and "rows" in answer:
                st.dataframe([dict(zip(answer["columns"], row)) for row in answer["rows"]])
            elif isinstance(answer, dict) and "error" in answer:
                st.error(answer["error"])
            else:
                st.markdown(answer)

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
            logo_img_url = "https://cdn-icons-png.flaticon.com/512/465/465077.png"
            st.sidebar.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <img src="{logo_img_url}" alt="icon" style="width:32px; height:32px; margin-right:10px;">
                    <h1 style="margin: 0; font-size: 28px;">Super Wheels</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.sidebar.markdown("""
            <style>
            div.stButton > button {
                width: 200px; 
                height: 48px; 
                font-size: 20px;  
                font-weight: 600;
                border-radius: 10px;
                margin-bottom: 16px; 
           
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            div.stButton > button:hover {
                background-color: rgba(255,255,255,0.1); 
            }
            </style>
            """, unsafe_allow_html=True)

            if st.sidebar.button("Dashboard"):
                st.session_state.view = "Dashboard"
            if st.sidebar.button("Chatbot"):
                st.session_state.view = "Chatbot"

            if "view" not in st.session_state:
                st.session_state.view = "Dashboard"

            if st.session_state.view == "Dashboard":
                admin_dashboard()
            else:
                chatbot()
        else:
            chatbot()
