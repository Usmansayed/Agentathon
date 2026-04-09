# Agentathon — Problem Statement
## Dept. of AI & Data Science | Ramaiah Institute of Technology

### Context
You are building an autonomous AI agent for RetailIQ, a fictional e-commerce analytics company.
The company has given you a dataset of customer orders. Your agent must analyze it, clean it,
and answer 5 business questions — completely autonomously, without any human intervention.

### What you are given
- train_data.csv — 3,500 rows of order data (your training/development data)
- This problem statement

### What will happen at the 3-hour mark
- You will receive test_data.csv — 1,500 new rows of order data your agent has never seen
- You must run your agent on this new file by only changing the file path
- You have 15 minutes to submit team-name.txt
- No code changes allowed after test data is released

### The 5 questions your agent must answer (applied to test data)
Q1: What is the total revenue per product category? Rank highest to lowest and
report values. Revenue = quantity × unit_price ×
(1 − discount_percent/100)
Q2: Which customer region has the highest average delivery time? Rank all
regions and report values.
Q3: How many data quality issues exist in the raw data? Count separately:
duplicate order IDs, quantity outliers (>1000), price format errors, invalid
discounts, total null cells.
Q4: What is the return rate (%) per payment method? Rank highest to lowest and
report values.
Q5: Write a 3-sentence executive summary of the business health based on the
data.

### Output format (MANDATORY)
Your agent must produce a plain text file called team-name.txt with EXACTLY
this format:

Q1: [your answer]
Q2: [your answer]
Q3: [your answer]
Q4: [your answer]
Q5: [your answer]

Submissions not in this format will score zero on auto-graded questions.

### Submission notes
- Keep labels in the same order: Q1, Q2, Q3, Q4, Q5.
- For Q1, Q2, and Q4 include ranked labeled values.
- For Q3 include all five counts.
- For Q5 write exactly 3 sentences.

### Recommended frameworks
- CrewAI: pip install crewai
- AutoGen: pip install pyautogen
- Google ADK: pip install google-adk
- LangGraph: pip install langgraph

### Rules
1. You may use any LLM API (bring your own key or use provided keys)
2. Internet access is allowed during the build phase
3. After test data is released: no code changes, no model calls outside your agent
4. One submission per team. Last submission before deadline counts.
5. Teams must be able to explain their architecture during the 5-minute demo

Good luck and build boldly.
