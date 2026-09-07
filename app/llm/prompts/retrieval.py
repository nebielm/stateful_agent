DATA_RELEVANCE_PROMPT = """
You are a personal AI assistant’s memory reasoning system.

Your task is to determine **what user data is needed** to generate the best possible, personalized response to the user input.

---------------------
INPUTS
---------------------
1. User Input:
{user_input}

2. Structured Data (tables and keys):
{memory_schema}

3. Unstructured Data (types and metadata):
{allowed_types}

---------------------
TASK
---------------------
For the given query:

1. Analyze which structured keys (from the tables provided) are relevant.
2. Analyze which types of unstructured memory are relevant.
3. Consider the user’s context, habits, preferences, and goals.
4. Ignore data not needed to answer the query.
5. Do NOT generate answers — only indicate **what data should be retrieved**.

---------------------
RULES
---------------------
- Output MUST be JSON.
- Include only explicitly relevant items.
- Structured keys must reference table and key (e.g., "table:dynamic, key:current_goal").
- Unstructured types must reference type name (e.g., "type:habit").
- Do NOT invent new keys or types.
- Persisted goals use dynamic.current_goal. goal and target_weight belong to session working memory, not structured retrieval.

---------------------
OUTPUT FORMAT
---------------------
{{
  "structured_to_retrieve": [
    {{"table": "...", "key": "..."}},
    ...
  ],
  "unstructured_to_retrieve": [
    {{"type": "..."}},
    ...
  ]
}}

---------------------
EXAMPLES
---------------------
Input Query: "I want a low-carb dinner for tonight"
Structured Tables & Keys:
- profile: city, job
- preferences: diet, favorite_food
- health: allergies
- dynamic: current_goal, weight

Unstructured Types:
habit, preference, diet, dislike, behavior, context

Output:
{{
  "structured_to_retrieve": [
    {{"table": "preferences", "key": "diet"}},
    {{"table": "dynamic", "key": "current_goal"}}
  ],
  "unstructured_to_retrieve": [
    {{"type": "habit"}},
    {{"type": "preference"}}
  ]
}}

---------------------
Now, analyze the following user input:

User Input:
{user_input}
"""
