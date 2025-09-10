You are performing **Hierarchical Text Classification (HTC)** of a given text.
The text is a Question and Answer Pair, which is followed by an Aspect and an Opinion extracted from the Question and Answer Pair. 
Your goal is to decide which child category best matches the given aspect, while considering the full context (Question, Answer, and Opinion).
You are currently at the parent category: **{{ current_label }}**.



Here is the text to classify:

---
Question: {{question}}
Answer: {{answer}}
Aspect: {{aspect}} 
Opinion: {{opinion}}
---

Candidate sub-categories (choose exactly one):
{{ candidate_list }}

Notes:
- If none of the categories fit, choose the option labeled **"null"**.
- Output **only the number** of the chosen option, nothing else.

{{ demos_block }}
