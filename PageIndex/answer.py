import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
TREE_FILE = "./results/cv_structure.json"
# -----------------------------``
# Load tree
# -----------------------------
def load_tree():
    with open(TREE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["structure"]
# -----------------------------
# Remove fields (like PageIndex utils.remove_fields)
# -----------------------------
def remove_fields(obj, fields):
    if isinstance(obj, dict):
        return {
            k: remove_fields(v, fields)
            for k, v in obj.items()
            if k not in fields
        }

    if isinstance(obj, list):
        return [remove_fields(x, fields) for x in obj]

    return obj
# -----------------------------
# Create node_id → node mapping
# -----------------------------
def create_node_mapping(tree):
    node_map = {}
    def walk(node):
        if not isinstance(node, dict):
            return
        node_id = node.get("node_id")
        if node_id:
            node_map[node_id] = node
        for child in node.get("nodes", []):
            walk(child)
    for root in tree:
        walk(root)
    return node_map
# -----------------------------
# Extract JSON from LLM output
# -----------------------------
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON returned by model:\n" + text)
    return json.loads(match.group())
# -----------------------------
# LLM Tree Search
# -----------------------------
def tree_search(query, tree):
    tree_without_text = remove_fields(tree, ["text"])
    prompt = f"""
You are given a question and a tree structure of a document.
Each node contains:
- node_id
- title
- summary
Find nodes that likely contain the answer.
Question:
{query}
Document tree:
{json.dumps(tree_without_text, ensure_ascii=False)[:15000]}
Reply ONLY JSON:
{{
 "thinking": "...",
 "node_list": ["node_id"]
}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content.strip()
    return extract_json(content)
# -----------------------------
# Retrieve context
# -----------------------------
def get_context(node_list, node_map):
    texts = []
    for node_id in node_list:
        node = node_map.get(node_id)
        if not node:
            continue
        text = node.get("text")
        if text:
            texts.append(text)
    return "\n\n".join(texts)
# -----------------------------
# Answer generation
# -----------------------------
def generate_answer(query, context):
    prompt = f"""
Answer the question based ONLY on the context.
Question:
{query}
Context:
{context}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
# -----------------------------
# Main pipeline
# -----------------------------
def main():
    query = "what does he use vuejs for"
    tree = load_tree()
    node_map = create_node_mapping(tree)
    # Step 1: tree search
    search_result = tree_search(query, tree)
    print("\nReasoning:\n")
    print(search_result["thinking"])
    print("\nRetrieved nodes:\n")
    for node_id in search_result["node_list"]:
        node = node_map[node_id]
        print(f"{node_id} | {node.get('title')}")
    # Step 2: retrieve context
    context = get_context(search_result["node_list"], node_map)
    print("\nContext preview:\n")
    print(context[:800], "...\n")
    # Step 3: generate answer
    answer = generate_answer(query, context)
    print("\nAnswer:\n")
    print(answer)
if __name__ == "__main__":
    main()

