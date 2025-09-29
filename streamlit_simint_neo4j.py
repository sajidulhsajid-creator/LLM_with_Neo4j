import os, re, time, sqlite3
import streamlit as st
from neo4j import GraphDatabase
from google import genai
from pyvis.network import Network
import streamlit.components.v1 as components

# ---------- Config & Secrets ----------
GENAI_API_KEY = os.getenv("GENAI_API_KEY") or st.secrets.get("GENAI_API_KEY")
NEO4J_URI  = os.getenv("NEO4J_URI") or st.secrets.get("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") or st.secrets.get("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS") or st.secrets.get("NEO4J_PASS")

genai_client = genai.Client(api_key=GENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ---------- SQLite History ----------
DB_PATH = "simint_llm_history.db"
def init_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_question TEXT NOT NULL,
            cypher TEXT NOT NULL,
            rows_count INTEGER,
            duration_ms INTEGER,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
def add_history(question, cypher, rows_count, duration_ms):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO history (user_question,cypher,rows_count,duration_ms) VALUES (?,?,?,?)",
                     (question, cypher, rows_count, duration_ms))
def get_history(limit=50):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, ts, user_question, cypher, rows_count, duration_ms FROM history ORDER BY id DESC LIMIT ?", (limit,))
        return cur.fetchall()
def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM history")

init_history()

# ---------- LLM: English -> Cypher ----------
SCHEMA_TEXT = """
(:Event)-[:HAS_IDEOLOGY]->(:Ideology)
(:Event)-[:HAS_TACTICAL_ACTOR]->(:TacticalActor)
(:Event)-[:HAS_TACTIC_MO]->(:TacticMO)
(:Event)-[:HAS_OPERATIONAL_TARGET]->(:OperationalTarget)
(:Event)-[:HAS_TYPE]->(:Type)
(:Event)-[:HAS_SOURCE]->(:Source)
(:Event)-[:ANALYZED_BY]->(:Analyst)
(:Event)-[:OCCURRED_IN]->(:Location)

Event props: id, title, date, summary, url, typeId, ideologyIds, tacticalActorIds, tacticIds, operationalTargetIds, locationIds
Location props: name, country
All others: name
"""

SYS_RULES = """
Return ONLY a single read-only Cypher statement.
- Disallow any write ops: CREATE, MERGE, DELETE, DETACH, SET, REMOVE, CALL dbms.*, APOC writes, LOAD CSV, PERIODIC COMMIT.
- Prefer MATCH with WHERE and RETURN.
- If user did not specify a limit, add LIMIT 100.
- Use the schema above; if in doubt, filter by properties that exist (e.g., Location.name, Location.country, Event.date).
- For graph visualization, return nodes and relationships using patterns like: MATCH (a)-[r]->(b) RETURN a, r, b
- For tabular data, return properties like: RETURN a.name, a.date
- NEVER mix node objects with properties in the same RETURN (e.g., don't do "RETURN a, a.name" - choose either "RETURN a" or "RETURN a.name")
- Output ONLY raw Cypher (no commentary, no markdown).
"""

def ask_gemini_for_cypher(user_q: str, prefer_graph: bool = False, limit: int = 20) -> str:
    graph_hint = ""
    if prefer_graph:
        graph_hint = f"\nIMPORTANT: For graph visualization, return nodes and relationships (e.g., MATCH (e:Event)-[r]->(l:Location) RETURN e, r, l LIMIT {limit}). Do NOT mix nodes with properties. Keep results under {limit} for readability."
    else:
        graph_hint = f"\nIMPORTANT: For tabular data, return only properties (e.g., RETURN e.title, e.date, l.name LIMIT {limit}). Do NOT return node objects."
    
    prompt = f"""You are a Cypher expert.
Schema:
{SCHEMA_TEXT}

Rules:
{SYS_RULES}{graph_hint}

User question: "{user_q}"
Cypher:"""
    resp = genai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = getattr(resp, "text", "") or ""
    # strip code fences if any
    text = re.sub(r"```(?:cypher)?\s*|\s*```", "", text, flags=re.I).strip()
    return text

# ---------- Safety: read-only guard ----------
WRITE_TOKENS = r"(?:CREATE|MERGE|DELETE|DETACH|SET|REMOVE|LOAD\s+CSV|CALL\s+dbms\.|CALL\s+apoc\.\w+\.write)"
def sanitize_cypher(cypher: str, default_limit=20) -> str:
    up = re.sub(r"\s+", " ", cypher.strip(), flags=re.S)
    if re.search(WRITE_TOKENS, up, flags=re.I):
        raise ValueError("Write operations are not allowed in read-only mode.")
    # Enforce single statement
    if ";" in up:
        raise ValueError("Multiple statements are not allowed.")
    
    # Fix common Neo4j syntax issues
    # Check if mixing nodes with properties in RETURN clause
    return_match = re.search(r'RETURN\s+(.+?)(?:\s+LIMIT|\s*$)', up, re.I)
    if return_match:
        return_clause = return_match.group(1)
        # Look for patterns like "a, a.property" or "a.property, a"
        if re.search(r'\b(\w+)\s*,.*\1\.\w+|\b(\w+\.\w+)\s*,.*\b\w+\s*(?!\.)', return_clause):
            # Try to fix by removing standalone node references
            fixed_return = re.sub(r',\s*\w+(?!\.)(?=\s*(?:,|$))', '', return_clause)
            fixed_return = re.sub(r'^\s*\w+(?!\.)\s*,\s*', '', fixed_return)
            up = up.replace(return_clause, fixed_return)
    
    # Add LIMIT if missing
    if not re.search(r'\bLIMIT\s+\d+\b', up, flags=re.I):
        up = f"{up} LIMIT {default_limit}"
    return up

# ---------- Neo4j run ----------
def run_cypher(query: str):
    with driver.session() as session:
        result = session.run(query)
        # Convert to records that preserve Neo4j types
        records = []
        for record in result:
            records.append(record)
        return records

# ---------- Enhanced Graph render ----------
def render_graph(records):
    if not records:
        st.info("No data to visualize in graph format.")
        return
        
    net = Network(height="700px", width="100%", bgcolor="#1e1e1e", font_color="white")
    
    # Much better physics settings for less congestion
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 200},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.1,
          "springLength": 300,
          "springConstant": 0.02,
          "damping": 0.09,
          "avoidOverlap": 1
        }
      },
      "layout": {
        "improvedLayout": true
      },
      "nodes": {
        "font": {
          "size": 14,
          "color": "white"
        },
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "font": {
          "size": 12,
          "color": "white"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.8
          }
        },
        "shadow": true,
        "smooth": {
          "enabled": true,
          "type": "continuous"
        }
      }
    }
    """)
    
    node_ids = set()
    edges_added = set()
    node_counts = {}  # Count nodes by type for sizing
    
    # First pass: count node types
    for record in records:
        for key, value in record.items():
            if hasattr(value, 'labels') and hasattr(value, 'id'):
                node_type = list(value.labels)[0] if value.labels else "Unknown"
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
    
    for record in records:
        for key, value in record.items():
            # Handle Neo4j Node objects
            if hasattr(value, 'labels') and hasattr(value, 'id'):
                node_id = value.id
                if node_id not in node_ids:
                    # Get node label - prefer 'name' property, fall back to labels
                    name = value.get('name', value.get('title', ''))
                    
                    # Truncate long labels to reduce congestion
                    if len(name) > 30:
                        label = name[:27] + "..."
                    else:
                        label = name
                    
                    if not label:  # fallback
                        label = f"{list(value.labels)[0] if value.labels else 'Node'}"
                    
                    # Create tooltip with key properties only
                    props = dict(value)
                    node_type = list(value.labels)[0] if value.labels else "Unknown"
                    
                    # Show only the most relevant properties in tooltip
                    key_props = {}
                    for prop in ['name', 'title', 'date', 'country']:
                        if prop in props and props[prop]:
                            key_props[prop] = props[prop]
                    
                    tooltip = f"{node_type}\n" + "\n".join([f"{k}: {v}" for k, v in key_props.items()])
                    
                    # Color and size by node type
                    color = get_node_color(node_type)
                    size = get_node_size(node_type, node_counts.get(node_type, 1))
                    
                    net.add_node(
                        node_id, 
                        label=str(label), 
                        title=tooltip, 
                        color=color,
                        size=size,
                        mass=2 if node_type == "Event" else 1  # Events have more mass
                    )
                    node_ids.add(node_id)
            
            # Handle Neo4j Relationship objects
            elif hasattr(value, 'type') and hasattr(value, 'start_node') and hasattr(value, 'end_node'):
                start_id = value.start_node.id
                end_id = value.end_node.id
                edge_key = (start_id, end_id, value.type)
                
                if edge_key not in edges_added:
                    # Add nodes if not already added
                    for node in [value.start_node, value.end_node]:
                        if node.id not in node_ids:
                            name = node.get('name', node.get('title', ''))
                            if len(name) > 30:
                                label = name[:27] + "..."
                            else:
                                label = name
                            
                            if not label:
                                label = f"{list(node.labels)[0] if node.labels else 'Node'}"
                                
                            props = dict(node)
                            node_type = list(node.labels)[0] if node.labels else "Unknown"
                            
                            # Show only key properties
                            key_props = {}
                            for prop in ['name', 'title', 'date', 'country']:
                                if prop in props and props[prop]:
                                    key_props[prop] = props[prop]
                            
                            tooltip = f"{node_type}\n" + "\n".join([f"{k}: {v}" for k, v in key_props.items()])
                            color = get_node_color(node_type)
                            size = get_node_size(node_type, node_counts.get(node_type, 1))
                            
                            net.add_node(
                                node.id, 
                                label=str(label), 
                                title=tooltip, 
                                color=color, 
                                size=size,
                                mass=2 if node_type == "Event" else 1
                            )
                            node_ids.add(node.id)
                    
                    # Add edge with cleaner styling
                    net.add_edge(
                        start_id, 
                        end_id, 
                        label=value.type.replace("_", " "), 
                        title=value.type,
                        length=250,  # Longer edges for less congestion
                        width=2
                    )
                    edges_added.add(edge_key)
            
            # Handle Path objects
            elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                for node in value.nodes:
                    if node.id not in node_ids:
                        name = node.get('name', node.get('title', ''))
                        if len(name) > 30:
                            label = name[:27] + "..."
                        else:
                            label = name
                            
                        if not label:
                            label = f"{list(node.labels)[0] if node.labels else 'Node'}"
                            
                        props = dict(node)
                        node_type = list(node.labels)[0] if node.labels else "Unknown"
                        
                        # Show only key properties
                        key_props = {}
                        for prop in ['name', 'title', 'date', 'country']:
                            if prop in props and props[prop]:
                                key_props[prop] = props[prop]
                        
                        tooltip = f"{node_type}\n" + "\n".join([f"{k}: {v}" for k, v in key_props.items()])
                        color = get_node_color(node_type)
                        size = get_node_size(node_type, node_counts.get(node_type, 1))
                        
                        net.add_node(
                            node.id, 
                            label=str(label), 
                            title=tooltip, 
                            color=color, 
                            size=size,
                            mass=2 if node_type == "Event" else 1
                        )
                        node_ids.add(node.id)
                
                for rel in value.relationships:
                    edge_key = (rel.start_node.id, rel.end_node.id, rel.type)
                    if edge_key not in edges_added:
                        net.add_edge(
                            rel.start_node.id, 
                            rel.end_node.id, 
                            label=rel.type.replace("_", " "), 
                            title=rel.type,
                            length=250,
                            width=2
                        )
                        edges_added.add(edge_key)

    if len(node_ids) == 0:
        st.warning("No graph nodes found. The query may be returning tabular data instead of graph relationships.")
        st.info("Try queries like: 'Show me the relationship graph between events and locations' or modify your query to return nodes and relationships.")
        return
    
    # Save and render
    net_path = "temp_graph.html"
    net.save_graph(net_path)
    
    try:
        with open(net_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=650)
        st.success(f"Graph rendered with {len(node_ids)} nodes and {len(edges_added)} relationships")
    except Exception as e:
        st.error(f"Error rendering graph: {e}")

def get_node_color(label):
    """Assign colors based on node labels"""
    colors = {
        "Event": "#ff6b6b",
        "Location": "#4ecdc4", 
        "Ideology": "#45b7d1",
        "TacticalActor": "#96ceb4",
        "TacticMO": "#ffeaa7",
        "OperationalTarget": "#dda0dd",
        "Type": "#98d8c8",
        "Source": "#fdcb6e",
        "Analyst": "#6c5ce7"
    }
    return colors.get(label, "#95a5a6")

def get_node_size(label, count):
    """Assign sizes based on node labels and frequency"""
    base_sizes = {
        "Event": 25,
        "Location": 35, 
        "Ideology": 30,
        "TacticalActor": 28,
        "TacticMO": 25,
        "OperationalTarget": 30,
        "Type": 25,
        "Source": 20,
        "Analyst": 20
    }
    base_size = base_sizes.get(label, 25)
    
    # Scale based on frequency (but not too much to avoid huge nodes)
    if count > 10:
        return min(base_size + 15, 50)
    elif count > 5:
        return base_size + 8
    else:
        return base_size

# ---------- Convert records for table display ----------
def records_to_table_data(records):
    """Convert Neo4j records to simple dict format for table display"""
    table_data = []
    for record in records:
        row = {}
        for key, value in record.items():
            if hasattr(value, 'labels') and hasattr(value, 'id'):  # Neo4j Node
                # Extract key properties for table view
                props = dict(value)
                # Show the most relevant property or a readable identifier
                display_value = (props.get('name') or 
                               props.get('title') or 
                               props.get('summary', '')[:50] + '...' if props.get('summary') else
                               f"{list(value.labels)[0] if value.labels else 'Node'}_{value.id}")
                row[f"{key} ({list(value.labels)[0] if value.labels else 'Node'})"] = display_value
            elif hasattr(value, 'type') and hasattr(value, 'start_node'):  # Neo4j Relationship
                row[f"{key} (Relationship)"] = value.type
            else:
                row[key] = value
        table_data.append(row)
    return table_data

    # Add graph controls and warnings
    if len(records) > 30:
        st.error(f"⚠️ Too many results ({len(records)} records) for clean visualization! Try:")
        st.write("- Add specific filters (country, date range, event type)")
        st.write("- Use smaller limits (5-20 items)")  
        st.write("- Switch to Table view for large datasets")
        
    elif len(records) > 15:
        st.warning(f"Large dataset ({len(records)} records). Graph may be cluttered.")
    
    # Add graph controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Re-layout", help="Restart graph layout"):
            pass  # Re-render will trigger new layout
    with col2:
        if st.button("📊 Stats"):
            node_type_counts = {}
            for record in records:
                for key, value in record.items():
                    if hasattr(value, 'labels') and hasattr(value, 'id'):
                        node_type = list(value.labels)[0] if value.labels else "Unknown"
                        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            
            st.write("**Graph Statistics:**")
            for node_type, count in sorted(node_type_counts.items()):
                st.write(f"- {node_type}: {count} nodes")
    with col3:
        if st.button("💡 Tips"):
            st.info("""
            **For cleaner graphs:**
            - Use LIMIT 5-15 in queries
            - Filter by specific locations/dates
            - Focus on one relationship type
            - Try: "Show 8 events in [country] with locations"
            """)
    
    # ---------- UI ----------
st.set_page_config(page_title="SIMINT • Neo4j (Natural Language)", layout="wide")
st.title("SIMINT • Neo4j Natural Language Query")

colL, colR = st.columns([2,1], gap="large")

with colR:
    st.subheader("History")
    hist = get_history(limit=100)
    if st.button("Clear history", use_container_width=True):
        clear_history()
        st.toast("History cleared", icon="🧹")
        st.rerun()
    if hist:
        for _id, ts, q, c, rows, ms in hist:
            with st.container(border=True):
                st.caption(ts)
                st.write(f"**Q:** {q}")
                with st.expander("View Cypher"):
                    st.code(c, language="cypher")
                cols = st.columns([1,1,1])
                cols[0].write(f"Rows: **{rows or 0}**")
                cols[1].write(f"Time: **{ms} ms**")
                if cols[2].button("Re-run", key=f"rerun_{_id}"):
                    st.session_state["prefill"] = q
                    st.rerun()
    else:
        st.info("No history yet. Run your first query!")

with colL:
    user_q = st.text_input(
        "Ask a question (Ctrl/⌘+Enter to run)",
        value=st.session_state.pop("prefill", "") if "prefill" in st.session_state else "",
        placeholder="e.g., Show 10 events in France with their locations"
    )
    
    # Query options
    col1, col2 = st.columns([2, 1])
    with col1:
        query_type = st.radio("Query type:", ["Auto", "Table view", "Graph view"], horizontal=True)
    with col2:
        limit_nodes = st.number_input("Max results for graphs:", min_value=5, max_value=100, value=20, step=5, help="Limit results to avoid congestion")
    
    run = st.button("Run Query (Neo4j)")
    
    st.write("""
    **Tips for CLEAN graph visualization:**
    - "Show 10 events in France with their locations" 
    - "Display 5 recent events and their ideologies"
    - "Graph 15 events in Kenya with connections" (always specify small numbers!)
    
    **Tips for table view:**
    - "List 50 events in France with title and date"
    - "Top 20 ideologies by name"
    - "Show all event details for 2023"
    
    ⚠️ **For readable graphs: Use small limits (5-20 items)!**
    """)

    if run and user_q.strip():
        prefer_graph = query_type == "Graph view" or (query_type == "Auto" and any(word in user_q.lower() for word in ["graph", "relationship", "connected", "network", "visualize"]))
        
        try:
            with st.spinner("Generating Cypher (Gemini)…"):
                raw = ask_gemini_for_cypher(user_q, prefer_graph=prefer_graph, limit=limit_nodes)
                cypher = sanitize_cypher(raw, default_limit=limit_nodes)
                st.code(cypher, language="cypher")
        except Exception as e:
            st.error(f"LLM/Cypher error: {e}")
        else:
            try:
                t0 = time.time()
                records = run_cypher(cypher)
                ms = int((time.time() - t0) * 1000)
                st.success(f"Query OK — {len(records)} records in {ms} ms")

                if records:
                    # Tabbed display: table + graph
                    tabs = st.tabs(["Table", "Graph"])
                    
                    with tabs[0]:
                        table_data = records_to_table_data(records)
                        if table_data:
                            st.dataframe(table_data)
                        else:
                            st.info("No tabular data to display")
                    
                    with tabs[1]:
                        render_graph(records)
                else:
                    st.info("No results returned")

                add_history(user_q, cypher, len(records), ms)
            except Exception as e:
                st.error(f"Neo4j query error: {e}")

# Example queries section
st.sidebar.subheader("Example Queries")

st.sidebar.write("**📊 Clean Graph Examples:**")
graph_examples = [
    "Show 10 events in France with locations",
    "Display 8 events and their ideologies", 
    "Graph 5 recent events with all connections",
    "Show 12 events in Kenya with locations",
    "Display 6 events connected to tactical actors"
]

for eq in graph_examples:
    if st.sidebar.button(eq, key=f"graph_{eq}"):
        st.session_state["prefill"] = eq
        st.rerun()

st.sidebar.write("**📋 Table Examples:**") 
table_examples = [
    "List 50 events in France with details",
    "Top 30 most common ideologies",
    "Show all locations by country",
    "Events from 2023 with sources",
    "List tactical actors and their events"
]

for eq in table_examples:
    if st.sidebar.button(eq, key=f"table_{eq}"):
        st.session_state["prefill"] = eq
        st.rerun()
