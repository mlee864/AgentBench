from src.agents.LLMCompiler.constants import END_OF_PLAN, END_OF_RESPONSE, JOINNER_FINISH, JOINNER_REPLAN

PLANNER_PROMPT = f"""You are an intelligent assistant designed to help plan sub-tasks to solve the user's question. 
You need to generate structured plan to solve a question answering task. 
Your role is to create detailed and efficient plans to solve a question answering task. 

### Tools Available:
- **search({{"text": "search text"}})**: Searches for 'search text' in the webshop environment. You can only use this tool when the observation explicitly shows a [search] button.
- **lookup({{page: "page to lookup from", keyword: "keyword to lookup"}}, [dependency])**: Lookup a specific keword from a page. Page should be the search query from the previous actions. Dependency denotes the actions need to be done before current action. For example, if you want to search about the color of apple, you need to search apple first and then lookup "color" from "apple" page. As the lookup can be done only after the search action, you need to specify dependency with the index of search action, in this case "$1". Dependency should be a list of "$number". The number is the index of dependent action.
1. search({{"text": "apple"}})
2. lookup({{"page": "apple", "keyword": "color"}}, ["$1"])
**"page" should be specific text from search history, do not use "$1", "$3", etc.**
- **join()**: Finalizes your plan so that the next agent can execute it and join the observations.

Here are some example plans:
"""
PLANNER_FEWSHOT_LIST=[
f"""Question: Are Pam Veasey and Jon Jost both American?
1. search({{"text": "Pam Veasey"}})
2. lookup({{"page": "Pam Veasey", "keyword": "American"}}, ["$1"])
3. search({{"text": "Jon Jost"}})
4. lookup({{"page": "Jon Jost", "keyword: "American"}}, ["$3"])
5. join(){END_OF_PLAN}
###
""",
f"""
Question: What is the profession of Bill Gates' mother and father?
1. search({{"text": "Bill Gates"}})
2. lookup({{"page": "Bill Gates", "keyword": "mother"}}, ["$1"])
3. lookup({{"page": "Bill Gates", "keyword": "father"}}, ["$2"])
4. search({{"text": "Bill Gates' mother"}})
5. lookup({{"page": "Bill Gates' mother", "keyword": "profession"}}, ["$3"])
6. search({{"text": "Bill Gates' father"}})
7. lookup({{"page": "Bill Gates' father", "keyword": "profession"}}, ["$5"])
8. join(){END_OF_PLAN}
###
""",
f"""
Question: What was the occupation of the mother of Albert Einstein?
1. search({{"text": "Albert Einstein"}})
2. lookup({{"page": "Albert Einstein", "keyword": "mother"}}, ["$1"])
3. join(){END_OF_PLAN}
###
""",
f"""
Question: What was the occupation of the mother of Albert Einstein? You need to search for Pauline Einstein.
1. search({{"text": "Pauline Einstein"}})
2. lookup({{"page": "Pauline Einstein", "keyword": "profession"}}, ["$1"])
3. lookup({{"page": "Pauline Einstein", "keyword": "occupation"}}, ["$1", "$2"])
4. join(){END_OF_PLAN}
###
""",
f"""
Question: What are the birthplaces of the parents of Charles Darwin?
1. search({{"text": "Charles Darwin"}})
2. lookup({{"page": "Charles Darwin", "keyword": "father"}}, ["$1"])
3. search({{"text": "Charles Darwin's father"}})
4. lookup({{"page": "Charles Darwin's father", "keyword": "birthplace"}}, ["$3"])
5. lookup({{"page": "Charles Darwin's father", "keyword": "birth"}}, ["$4"])
6. lookup({{"page": "Charles Darwin's father", "keyword": "born"}}, ["$5"])
7. lookup({{"page": "Charles Darwin", "keyword": "mother"}}, ["$1"])
8. search({{"text": "Charles Darwin's mother"}})
9. lookup({{"page": "Charles Darwin's mother", "keyword": "birthplace"}}, ["$8"])
10. lookup({{"page": "Charles Darwin's mother", "keyword": "birth"}}, ["$9"])
11. lookup({{"page": "Charles Darwin's mother", "keyword": "born"}}, ["$10"])
12. join(){END_OF_PLAN}
###
"""
]

OUTPUT_PROMPT = (
    "You must solve the Question. You are given Observations and you can use them to solve the Question. "
    "You MUST provide a Thought, and then an Action.\n"
    "Answer should always be a single item and MUST not be multiple choices.\n"
    "You will be given a question and some Wikipedia passages, which are the observations.\n\n"
    "Thought step can reason about the observations in 1-2 sentences, and Action can be only one type:\n"
    f" (1) {JOINNER_FINISH}(answer): returns the answer and finishes the task. "
    f" (2) {JOINNER_REPLAN}(reflection): if the provided observations are not enough to answer the question, you need to return {JOINNER_REPLAN}(reflection) with reflection. Reflection should include your reflection on current state, direction to next plan and what information you need to answer the question."
    "\n"
    "Follow the guidelines that you will die if you don't follow:\n"
    "  - Answer should be short and a single item and MUST not be multiple choices.\n"
    "  - Thought should be 1-2 sentences.\n"
    f"  - You must say {END_OF_RESPONSE} at the end of your response.\n"
    f"  - Do not answer using your internal knowledge. Only answer based on observations. If the observations are not enough, feel free to return {JOINNER_REPLAN} with reflection, then the system will give you further observations.\n"
    f"  - Answer and reflection of {JOINNER_FINISH}(answer) and {JOINNER_REPLAN}(reflection) should be passes as a argument. For example: \n"
    f"Action: {JOINNER_FINISH}(yes)\n"
    f"Action: {JOINNER_REPLAN}(I need to search for Bronisława Skłodowska's birthplace.)\n"
    "\n"
    "Here are some examples:\n"
    "\n"
)

OUTPUT_FEWSHOT_LIST=[
f"""Question: Which magazine was started first Arthur's Magazine or First for Women?
search(Arthur's Magazine)
Observation: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
search(First for Women)
Observation: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought: Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so ArthurMagazine was started first.\n"
"Action: {JOINNER_FINISH}(Arthur's Magazine)
{END_OF_RESPONSE}
""",
f"""
Question: What is the birth date of the father of the founder of Microsoft?
search(Bill Gates)
Observation: Bill Gates (born October 28, 1955) is an American business magnate, software developer, and co-founder of Microsoft Corporation.
search(Bill Gates' father)
Observation: Bill Gates' father is William H. Gates Sr., an American attorney and philanthropist.
Thought: We know Bill Gates' father's name is William H. Gates Sr., but his birth date is not yet available.
Action: {JOINNER_REPLAN}(I need to search for William H. Gates Sr.'s birth date.)
{END_OF_RESPONSE}
""",
f"""
search(William H. Gates Sr.)
Observation: William H. Gates Sr. was a prominent attorney, philanthropist, and the father of Bill Gates. More details about his birth are needed.
lookup([{{'page': 'William H. Gates Sr.', 'keyword': 'born'}}])
Observation: William H. Gates Sr. was born on November 30, 1925.
Thought: William H. Gates Sr.'s birth date is November 30, 1925.
Action: {JOINNER_FINISH}(November 30, 1925)
{END_OF_RESPONSE}
""",
f"""
Question: What was the birthplace of the mother of Marie Curie?
search(Marie Curie)
Observation: Marie Curie (born November 7, 1867 – July 4, 1934) was a Polish-born physicist and chemist, famous for her pioneering research on radioactivity.
search(Marie Curie's mother)
Observation: Marie Curie's mother was Bronisława Skłodowska, a teacher and the mother of Marie Curie.
Thought: We know the name of Marie Curie's mother, Bronisława Skłodowska, but her birthplace is not provided.
Action: {JOINNER_REPLAN}(I need to search for Bronisława Skłodowska's birthplace.)
{END_OF_RESPONSE}
""",
f"""
search(Bronisława Skłodowska)
Observation: Bronisława Skłodowska was born in Warsaw, Poland, and was a teacher by profession.
lookup([{{'page': 'Bronisława Skłodowska', 'keyword': 'birthplace'}}])
Observation: Bronisława Skłodowska was born in Warsaw, Poland.
Thought: Bronisława Skłodowska was born in Warsaw, Poland.
Action: {JOINNER_FINISH}(Warsaw, Poland)
{END_OF_RESPONSE}
"""
]

def get_planner_prompt(fewshot: int):
    if fewshot > len(PLANNER_FEWSHOT_LIST):
        fewshot = len(PLANNER_FEWSHOT_LIST)
        print(f"Max fewshot examples for hotpotqa planner prompt is {len(PLANNER_FEWSHOT_LIST)}. Running with {fewshot} fewshot examples.")
    return PLANNER_PROMPT + "\n".join(PLANNER_FEWSHOT_LIST[:fewshot])

def get_output_prompt(fewshot: int):
    if fewshot > len(OUTPUT_FEWSHOT_LIST):
        fewshot = len(OUTPUT_FEWSHOT_LIST)
        print(f"Max fewshot examples for hotpotqa output prompt is {len(OUTPUT_FEWSHOT_LIST)}. Running with {fewshot} fewshot examples.")
    return OUTPUT_PROMPT + "\n".join(OUTPUT_FEWSHOT_LIST[:fewshot])