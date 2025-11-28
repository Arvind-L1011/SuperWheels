import streamlit as st
import io
import requests
import os
import mysql.connector
import bcrypt
import time
import matplotlib.pyplot as plt
import speech_recognition as sr
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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

def ask_groq(prompt, temperature=0.7, max_tokens=1024, system_message="You are a helpful assistant."):
    """
    Send the messages to Groq chat completions endpoint and return the assistant text.
    Returns a string on success or an error string starting with '[GROQ API ERROR]'.
    """
    if not GROQ_API_KEY:
        return "[GROQ API ERROR] Missing GROQ_API_KEY environment variable."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Build messages similar to OpenAI-style chat completions
    system_msg = {"role": "system", "content": system_message}
    user_msg = {"role": "user", "content": prompt}

    # Keep some conversation context from session chat_history if available
    history = [msg for msg in st.session_state.get("chat_history", []) if msg.get("content_type", "text") == "text"]
    messages = [system_msg] + [{"role": msg["role"], "content": msg["content"]} for msg in history] + [user_msg]

    # A rough token estimator (same approach as original)
    def count_tokens(text):
        return len(text) // 4

    token_budget = 7000
    total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
    while total_tokens > token_budget and len(messages) > 2:
        # pop the oldest assistant/user message after system
        messages.pop(1)
        total_tokens = sum(count_tokens(msg["content"]) for msg in messages)

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return f"[GROQ API ERROR] Request failed: {e}"

    if resp.status_code != 200:
        # include body text for debugging
        try:
            return f"[GROQ API ERROR] {resp.status_code} - {resp.text}"
        except:
            return f"[GROQ API ERROR] {resp.status_code} - (unable to read response)"
    try:
        j = resp.json()
    except Exception as e:
        return f"[GROQ API ERROR] Could not parse JSON response: {e}"

    # Try common response shapes; prefer choices[0].message.content (OpenAI-like)
    try:
        # new Groq endpoint is OpenAI-compatible: choices[0].message.content
        return j["choices"][0]["message"]["content"]
    except Exception:
        # fallback: choices[0].text
        try:
            return j["choices"][0]["text"]
        except Exception:
            # last resort: full JSON string
            return f"[GROQ API ERROR] Unexpected response shape: {j}"

# ---------------------------
# Replace original ask_together usage:
def ask_together(prompt):
    """
    Backwards-compat shim: to minimize other code changes, keep same name.
    Internally calls ask_groq.
    """
    return ask_groq(prompt)

#-------------------------------

def typing_effect(text, speed=0.005):
    """Displays text with a typing animation."""
    placeholder = st.empty()
    displayed_text = ""

    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(speed)

    return placeholder

#--------------------------------
import pyttsx3
import tempfile
import base64

def generate_tts_audio(text):
    """
    Convert text to base64 WAV audio fully offline using pyttsx3.
    Returns base64 audio string or None on failure.
    """
    try:
        engine = pyttsx3.init()

        # Increase speed a bit (optional)
        rate = engine.getProperty("rate")
        engine.setProperty("rate", rate + 10)

        # Temporary wav file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_name = tmp.name
        tmp.close()

        engine.save_to_file(text, tmp_name)
        engine.runAndWait()

        with open(tmp_name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()

        return audio_base64

    except Exception as e:
        print("TTS ERROR:", e)
        return None

def render_tts_button(audio_b64, key):
    """Renders Speak button next to Download button on same line."""
    html_code = f"""
        <div style="display:flex; align-items:center; gap:12px; margin-top:6px;">

            <audio id="tts_audio_{key}">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>

            <button id="tts_btn_{key}"
                style="
                    background:#2b82f6;
                    color:white;
                    border:none;
                    padding:6px 14px;
                    border-radius:6px;
                    cursor:pointer;
                    font-size:14px;
                "
                onclick="
                    var audio = document.getElementById('tts_audio_{key}');
                    var btn = document.getElementById('tts_btn_{key}');
                    if (audio.paused) {{
                        audio.play();
                        btn.innerText='⛔ Stop';
                    }} else {{
                        audio.pause();
                        audio.currentTime = 0;
                        btn.innerText='🔊 Speak';
                    }}
                ">
                🔊 Speak
            </button>

            <script>
                var audio = document.getElementById("tts_audio_{key}");
                var btn = document.getElementById("tts_btn_{key}");
                audio.onended = function() {{
                    btn.innerText = "🔊 Speak";
                }};
            </script>
        </div>
    """

    st.components.v1.html(html_code, height=50)


#---------------------------------

# ---------------------------

def is_db_question(prompt):
    db_keywords = [
    "list", "show", "find", "get", "fetch", "display", "view", "retrieve", "mileage", "engine", "price", "brand", "model", "year", "manufacture", "color", "quantity", "available", "stock", "petrol", "diesel", "fuel", "engine type", "above", "below", "under", "over", "greater", "less", "between", "top", "lowest", "highest", "toyota", "honda", "tata", "hyundai", "mahindra", "kia", "carens", "creta", "nexon", "xuv", "city", "car", "cars", "vehicle", "vehicles", "models"]
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

        - When using aggregate functions, ALWAYS provide a clear alias name:
        - Never return unnamed aggregate columns.
        - The alias must be a single lowercase name with underscores.

        - Only return valid SQL statements. Do not include any explanation or prefix.
        - Return ONLY the final valid MySQL SELECT query.
        - Do not explain, describe, or annotate the query. 

        Now convert the following question to SQL:
        "{prompt}"
        """
        sql_query = ask_groq(sql_prompt)
        if isinstance(sql_query, str) and sql_query.startswith("[GROQ API ERROR]"):
            return {"error": sql_query}
        return execute_db_query(sql_query)
    else:
        return ask_groq(prompt)

def admin_dashboard():
    st.title("🧑‍💼 Admin Dashboard")
    tab_1,tab_2,tab_3,tab_4 = st.tabs(["User","Logs","Car Inventory", "Blocked Keywords"])

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

            col_1, col_2, col_3 = st.columns([4, 3, 5])
            col_1.markdown("**Email**")
            col_2.markdown("**Role**")

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
        
    with tab_4:
        st.subheader("Blocked Keywords Settings")

        st.markdown("These keywords will be blocked from user queries.")

        # Display existing keywords
        keywords = get_blocked_keywords()

        if keywords:
            st.write("### Blocked Keywords:")
            for i, kw in enumerate(keywords):
                col1, col2 = st.columns([6, 1])
                col1.write(f"- **{kw}**")
                if col2.button("Delete", key=f"del_kw_{i}"):
                    delete_blocked_keyword(kw)
                    st.success(f"Deleted keyword: {kw}")
                    st.rerun()
        else:
            st.info("No blocked keywords yet.")

        st.write("---")
        st.write("### ➕ Add New Keyword")

        new_kw = st.text_input("Keyword to block", key="new_kw_input")
        if st.button("Add Keyword"):
            if new_kw.strip() == "":
                st.warning("Keyword cannot be empty.")
            else:
                if add_blocked_keyword(new_kw.strip()):
                    st.success(f"Added keyword: {new_kw}")
                    st.rerun()
                else:
                    st.error("Keyword already exists or failed to insert.")



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

    import re
    
    if action == "Sign up":
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input("Email ID", placeholder="Enter Email", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="Enter Password", key="signup_password")

            submitted = st.form_submit_button("Create Account")

            if submitted:
                with st.spinner("Creating a new account..."):
                    if not email or not password:
                        st.warning("Please enter the credentials.")
                    elif not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{6,}$', password):
                        st.warning("Password must be at least 6 characters long and include:\n- 1 uppercase letter\n- 1 number\n- 1 special character.")
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

                            from datetime import datetime
                            from zoneinfo import ZoneInfo

                            log_conn = connect_db()
                            log_cursor = log_conn.cursor()
                            ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
                            log_cursor.execute("INSERT INTO login_logs (email, login_time) VALUES (%s, %s)",(email, ist_now))
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
    elements.append(Paragraph("Super Wheels Chatbot", styles["Title"]))
    elements.append(Spacer(1, 12))
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

def listen():
    """Listen to user's voice and return recognized text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #st.info("🎤 Listening... please speak your question clearly")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            #st.success(f"🗣 You said: {query}")
            return query
        except sr.UnknownValueError:
            st.warning("Sorry, I couldn't understand your voice. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None

# -----------------------------
# KEYWORD BLOCKER (DB managed)
# -----------------------------

def get_blocked_keywords():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT keyword FROM blocked_keywords")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0].lower() for r in rows]

def add_blocked_keyword(keyword):
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO blocked_keywords (keyword) VALUES (%s)", (keyword.lower(),))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except:
        return False

def delete_blocked_keyword(keyword):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM blocked_keywords WHERE keyword = %s", (keyword.lower(),))
    conn.commit()
    cur.close()
    conn.close()

BLOCKED_RESPONSE = (
    "⚠️ This question contains restricted content and cannot be processed. "
    "Please ask something else."
)

def check_blocked_keywords(user_text):
    keyword_list = get_blocked_keywords()
    txt = user_text.lower()
    return any(k in txt for k in keyword_list)


def add_message(role, content, content_type="text", columns=None, rows=None, graph=None):
    """
    Safely add a new message to chat history only if it's not already there.
    """
    message = {
        "role": role,
        "content_type": content_type,
        "content": content,
        "columns": columns,
        "rows": rows,
        "graph": graph,
    }

    # Avoid duplicates by checking last message
    if not st.session_state.chat_history or st.session_state.chat_history[-1] != message:
        st.session_state.chat_history.append(message)


# -------------------------
# Part 2 (UI & chatbot view)
# -------------------------
from streamlit_chat_widget import chat_input_widget
from streamlit_extras.bottom_container import bottom

def chatbot():
    st.title("🤖 Chatbot")

    # --- Custom CSS ---
    st.markdown("""
        <style>
        .main > div { padding-bottom: 120px !important; }
        .stBottomContainer {
            background-color: #0e1117 !important;
            border-top: 1px solid #444 !important;
            height: 80px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 10px 20px !important;
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            z-index: 5 !important;
            box-shadow: 0 -3px 8px rgba(0,0,0,0.4);
        }
        .stBottomContainer textarea, .stBottomContainer input {
            min-height: 45px !important;
            font-size: 16px !important;
            border-radius: 8px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Initialize session trackers ---
    if "last_processed_input" not in st.session_state:
        st.session_state.last_processed_input = None
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # --- Display only assistant + user messages (download only for assistant) ---
    if st.session_state.chat_history:
        for idx, msg in enumerate(st.session_state.chat_history):
            role = msg["role"]
            with st.chat_message(role):
                if msg.get("content_type") == "table":
                    st.dataframe([dict(zip(msg["columns"], row)) for row in msg["rows"]],
                                 use_container_width=True)
                    if msg.get("graph"):
                        buf = io.BytesIO(msg["graph"])
                        img = plt.imread(buf, format="png")
                        fig, ax = plt.subplots()
                        ax.imshow(img)
                        ax.axis("off")
                        st.pyplot(fig)
                    # --- Download only for assistant replies ---
                    if role == "assistant":
                        if not msg.get("pdf"):
                            msg["pdf"] = create_pdf_response("User", {"columns": msg["columns"], "rows": msg["rows"]})
                        st.download_button("📄 Download",
                                           data=msg["pdf"],
                                           file_name=f"chat_table_{idx}.pdf",
                                           mime="application/pdf",
                                           key=f"download_table_{idx}")
                elif msg.get("content_type") == "text":
                    st.markdown(msg["content"])
                    # --- Download only for assistant replies ---
                    if role == "assistant":
                        if not msg.get("pdf"):
                            msg["pdf"] = create_pdf_response("User", msg["content"])
                        st.download_button("📄 Download",
                                           data=msg["pdf"],
                                           file_name=f"chat_text_{idx}.pdf",
                                           mime="application/pdf",
                                           key=f"download_text_{idx}")

    # --- Bottom chat input widget ---
    with bottom():
        st.markdown(
            """ <style> iframe[title="streamlit_chat_widget.chat_input_widget"] 
            { 
            height:80px!important;
            width:100%!important;
            min-height:80px!important; 
            max-height:80px!important;
            border:none!important;
            overflow:hidden!important; 
            } 
            </style> 
            """, unsafe_allow_html=True)

        user_input = chat_input_widget()

    # --- Process new input safely ---
    if user_input and not st.session_state.processing:
        st.session_state.processing = True

        if "text" in user_input:
            question = user_input["text"].strip()
        else:
            question = None

        if question and question != st.session_state.last_processed_input:

            # ---- Keyword Blocker ----
            if check_blocked_keywords(question):
                with st.chat_message("assistant"):
                    st.markdown(BLOCKED_RESPONSE)

                add_message("assistant", BLOCKED_RESPONSE, "text")

                pdf_file = create_pdf_response(question, BLOCKED_RESPONSE)

                st.download_button(
                    "📄 Download",
                    data=pdf_file,
                    file_name="blocked_message.pdf",
                    mime="application/pdf",
                    key=f"blocked_download_{len(st.session_state.chat_history)}"
                )

                st.session_state.processing = False
                return


            st.session_state.last_processed_input = question

            # --- User message ---
            with st.chat_message("user"):
                st.markdown(question)
            add_message("user", question)

            # --- Assistant response ---
            with st.chat_message("assistant"):
                with st.spinner("Revving the engines...🚗💨"):
                    answer = ask_combined(question)

                # TABLE / GRAPH RESPONSE
                if isinstance(answer, dict) and "columns" in answer and "rows" in answer:
                    st.dataframe([dict(zip(answer["columns"], row)) for row in answer["rows"]],
                                 use_container_width=True)
                    fig, fig_bytes = None, None
                    if user_requested_graph(question):
                        fig = generate_bar_graph(answer["columns"], answer["rows"])
                        if fig:
                            st.pyplot(fig)
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png")
                            buf.seek(0)
                            fig_bytes = buf.read()

                    pdf_file = create_pdf_response(question, answer, fig)
                    add_message("assistant", "", "table", answer["columns"], answer["rows"], fig_bytes)

                    # ✅ Show download button instantly for current answer
                    st.download_button("📄 Download",
                                       data=pdf_file,
                                       file_name="query_response.pdf",
                                       mime="application/pdf",
                                       key=f"download_current_table_{len(st.session_state.chat_history)}")

                # ERROR RESPONSE
                elif isinstance(answer, dict) and "error" in answer:
                    st.error(answer["error"])
                    add_message("assistant", answer["error"], "text")
                    pdf_file = create_pdf_response(question, answer["error"])
                    st.download_button("📄 Download",
                                       data=pdf_file,
                                       file_name="error_response.pdf",
                                       mime="application/pdf",
                                       key=f"download_error_{len(st.session_state.chat_history)}")

                # TEXT RESPONSE
                else:
                    typing_effect(answer)

                    # --- PDF ---
                    pdf_file = create_pdf_response(question, answer)
                    add_message("assistant", answer, "text")

                    # --- Download Button (existing) ---
                    # Create inline container for Download + Speak button
                    btn_col = st.container()

                    with btn_col:
                        st.write(
                            f"""
                            <div style="display:flex; align-items:center; gap:14px; margin-top:6px;">
                                <div id="download_btn_{len(st.session_state.chat_history)}"></div>
                                <div id="tts_btn_wrap_{len(st.session_state.chat_history)}"></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # ---- DOWNLOAD BUTTON injected into first div ----
                    st.download_button(
                        "📄 Download",
                        data=pdf_file,
                        file_name="chat_response.pdf",
                        mime="application/pdf",
                        key=f"download_current_text_{len(st.session_state.chat_history)}"
                    )

                    # ---- TTS BUTTON injected into second div ----
                    audio_b64 = generate_tts_audio(answer)
                    if audio_b64:
                        render_tts_button(
                            audio_b64,
                            key=f"tts_{len(st.session_state.chat_history)}"
                        )


        st.session_state.processing = False


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
