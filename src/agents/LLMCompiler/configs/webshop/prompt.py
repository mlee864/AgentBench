from src.agents.LLMCompiler.constants import END_OF_PLAN, END_OF_RESPONSE

PLANNER_PROMPT = """You are an intelligent assistant designed to help plan sub-tasks to solve the user's question. 

The user requests assistance in purchasing specific products within a webshop environment. 
Your role is to create detailed and efficient plans to find, configure, and purchase products that meet the user's requirements.

### Tools Available:
- **search('search text')**: Searches for 'search text' in the webshop environment. You can only use this tool when the observation explicitly shows a [search] button.
- **click('button')**: Clicks a button to inspect product details, navigate pages, or interact with options. You are only allowed to click buttons displayed inside the [brackets].
- **join()**: Finalizes your plan so that the next agent can execute it and join the observations.

### Rules to Follow:
1. **Error Handling**: If the observation shows an Error message, refer to the current page to determine the next action. If you cannot decide the next action based on the valid observation, try clicking [Back to Search] to return to the search page or [< Prev] to go to the previous page.
2. **Step-by-Step and replan**: Carefully analyze each observation and ensure your actions align with the given information. You can replan after receiving the action results, so just plan for current state.
3. **Numbered Plans Only**: Present your plan as a numbered list, detailing each action step by step. Do not include explanations or extra text outside the numbered list.
4. **Navigating Pages**:
- After clicking Description, Features, Reviews, or Attributes, make sure to click '< Prev' to get back to the item page. 
- If you want to navigate another item after clicking specific item, you need to click '< Prev' first and then click the next item ID.

Your primary objective is to create actionable and concise plans that adhere to these rules, ensuring the user's requirements are met.

Here are example plans:
"""

PLANNER_FEWSHOT_LIST=[
f"""
1. search(search query)
2. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B011S76LB0] 
Item description for B011S76LB0
[B0096E5948]
Item description for B0096E5948
(Based on the search result,)

1. click(B011S76LB0)
2. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
[< Prev] 
Item description for B011S76LB0
Price: $21.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Attributes] 
[Buy Now] 
(If you want to check other products, you need to click '< Prev' to go back to search result page!)

1. click(< Prev)
2. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B011S76LB0] 
Item description for B011S76LB0
[B0096E5948]
Item description for B0096E5948
    
1. click(B0096E5948)
2. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
[< Prev] 
Item description for B0096E5948
Price: $21.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Attributes] 
[Buy Now] 
(Assuming B011S76LB0 is better item you think,)

1. click(< Prev)
2. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B011S76LB0] 
Item description for B011S76LB0
[B0096E5948]
Item description for B0096E5948
    
1. click(B011S76LB0)
2. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
[< Prev] 
Item description for B011S76LB0
Price: $21.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Attributes] 
[Buy Now] 
(If you are not sure about this item, you can inspect details of the product. Also, make sure to click '< Prev',)

1. click(Description)
2. click(< Prev)
3. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
[< Prev] 
Item description for B011S76LB0
Price: $21.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Attributes] 
[Buy Now]

1. click(Features)
2. click(< Prev)
3. join(){END_OF_PLAN}\n

Observation: 
[Back to Search] 
[< Prev] 
Item description for B011S76LB0
Price: $21.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Attributes] 
[Buy Now]
(After checking the details of items, click options and 'Buy Now')

1. click(option1)
2. click(option2)
3. click(Buy Now)
4. join(){END_OF_PLAN}\n
""",
f"""
Question: I am looking for dairy-free and apple-flavored snack bars with a variety pack priced lower than $30.
1. search('dairy-free apple snack bar variety pack under $30')
2. join(){END_OF_PLAN}\n

Thought: Based on the observations, the product with item ID B06Y96N1KG matches the criteria as it is dairy-free, apple-flavored, part of a variety pack, and priced at $21.49, which is under $30. I will select this item.
1. click('B06Y96N1KG')
2. join(){END_OF_PLAN}\n

Thought: Based on the observations, I will select specific options to meet my preferences.
1. click('Pack of 6')
2. click('Apple')
3. join(){END_OF_PLAN}\n

Thought: I have finalized the product configuration and will now proceed to purchase.
1. click('Buy Now')
2. join(){END_OF_PLAN}\n
###
""",
f"""
Question: Find the gluten-free chocolate protein bar under $20.
1. search('gluten-free chocolate protein bar under $20')
2. join(){END_OF_PLAN}\n

Thought: Based on the observations, the product with item ID B07GJTKYJQ matches the criteria as it is gluten-free, chocolate-flavored, priced at $19.99, and has a rating of 4.5 stars. I will select this item.
1. click('B07GJTKYJQ')
2. join(){END_OF_PLAN}\n

Thought: Based on the observations, I will select specific options such as flavor and quantity.
1. click('12 Bars')
2. click('Chocolate')
3. join(){END_OF_PLAN}\n

Thought: I have finalized the product configuration and will now proceed to purchase.
1. click('Buy Now')
2. join(){END_OF_PLAN}\n
###

Question: Find a vegan snack pack with a flavor variety under $15.
1. search('vegan snack pack flavor variety under $15')
2. join(){END_OF_PLAN}\n

Thought: Based on the observations, no suitable product was found. I will refine my search criteria to broaden the results.
1. click('< Back to Search')
2. search('vegan snack pack variety')
3. join(){END_OF_PLAN}\n

Thought: Based on the observations, the product with item ID B09XYZ123 matches the criteria as it is vegan, includes a flavor variety, and is priced at $14.99. I will select this item.
1. click('B09XYZ123')
2. join(){END_OF_PLAN}\n

Thought: Based on the observations, I will select specific options to meet my preferences and finalized the product configuration and will now proceed to purchase.
1. click('Variety Pack')
2. click('Flavor A')
3. click('Buy Now')
4. join(){END_OF_PLAN}\n
###
""",
]


OUTPUT_PROMPT = f"""
Solve a shopping task with interleaving Observation, Thought, and Action steps. Here are some guidelines:
- You will be given a Question and some observations from a web shop environment.
- Thought needs to reason about the question based on the Observations in 1-2 sentences.
- If the Observations are unclear, you must refine the query or navigate the environment until relevant data is found. You MUST NEVER say in your thought that you don't know the answer.
- You must say {END_OF_RESPONSE} at the end of your response.
  
Action can be one of the following types:
(1) Finish(answer): Returns the answer and finishes the task. Answer must be concise and specific. Answer MUST NEVER be 'unclear', 'unknown', or similar terms, or you will be PENALIZED.
(2) Replan(reason): Adjusts the approach based on observations to refine the search or explore further options. Answer must be concise and specific. Answer MUST NEVER be 'unclear', 'unknown', or similar terms, or you will be PENALIZED.

## Your answer should be
Thought: "your thought"
Action: "Replan or Finish"
{END_OF_RESPONSE}

Here are some examples:
"""

OUTPUT_FEWSHOT_LIST=[
f"""
Question: I am looking for dairy-free and apple-flavored snack bars with a variety pack priced lower than $30.
search('dairy-free apple snack bar variety pack under $30')
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96N1KG]
Dairy-Free Apple Snack Bars Variety Pack - $21.49

Thought: Based on the observations, the product with item ID B06Y96N1KG matches the criteria as it is dairy-free, apple-flavored, part of a variety pack, and priced at $21.49, which is under $30.
Action: Replan(Click "B06Y96N1KG" to view product details and proceed with configuration)
{END_OF_RESPONSE}

click('B06Y96N1KG')
Observation:
[Back to Search]
[< Prev]
Options:
flavor name [Apple][Strawberry]
size [Pack of 6][Pack of 12]
Price: $21.49

Thought: I will select the specific options 'Pack of 6' and 'Apple' to match my preferences.
Action: Replan(Click "Pack of 6" and "Apple" options to configure the product)
{END_OF_RESPONSE}

click('Pack of 6')
click('Apple')
Observation: You have clicked Pack of 6 and Apple.

Thought: I have finalized the product configuration and need to click 'Buy Now' to complete the purchase.
Action: Replan(Click 'Buy Now' to finalize the purchase)
{END_OF_RESPONSE}

click('Buy Now')
Observation: Your score (min 0.0, max 1.0): 0.5

Thought: Purchase successful with score 0.5
Action: Finish(Purchase successful with score 0.5)
{END_OF_RESPONSE}
###
""",
"""
Question: Find a vegan snack pack with a flavor variety under $15.
search('vegan snack pack flavor variety under $15')
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
No suitable product found.

Thought: I will refine my search to broaden the results.
Action: Replan(Click "Back to Search" to return to search and search again with broader criteria)
{END_OF_RESPONSE}

click('< Back to Search')
search('vegan snack pack variety')
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B09XYZ123]
Vegan Snack Pack Variety - $14.99

Thought: Based on the observations, the product with item ID B09XYZ123 matches the criteria as it is vegan, includes a flavor variety, and is priced at $14.99.
Action: Replan(Click "B09XYZ123" to view product details and proceed with configuration)

click('B09XYZ123')
Observation:
[Back to Search]
[< Prev]
Options:
flavor name [Flavor A][Flavor B]
size [Variety Pack]
Price: $14.99

Thought: I will select the specific options 'Variety Pack' and 'Flavor A' to match my preferences.
Action: Replan(Click "Variety Pack" and "Flavor A" options to configure the product)
{END_OF_RESPONSE}


1. click('Variety Pack')
2. click('Flavor A')
3. click('Buy Now')

Observation: Your score (min 0.0, max 1.0): 1.0

Thought: Purchase successful with score 1.0
Action: Finish(Purchase successful with score 1.0)
{END_OF_RESPONSE}
""",
"""
Question: Find a pair of wireless headphones under $100.
search('wireless headphones under $100')
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
No suitable product found.

Thought: I will refine my search to broaden the results.
Action: Replan(Click "Back to Search" to return to search and search again with broader criteria)
{END_OF_RESPONSE}

click('< Back to Search')
search('wireless headphones Bluetooth')
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B08XYZ123]
Wireless Bluetooth Headphones - $99.99

Thought: Based on the observations, the product with item ID B08XYZ123 matches the criteria as it is wireless, Bluetooth-enabled, and priced at $99.99.
Action: Replan(Click "B08XYZ123" to view product details and proceed with configuration)

click('B08XYZ123')
Observation:
[Back to Search]
[< Prev]
Options:
color [Black][White]
size [Standard]
Price: $99.99

Thought: I will select the specific options 'Standard' and 'Black' to match my preferences.
Action: Replan(Click "Standard" and "Black" options to configure the product)
{END_OF_RESPONSE}

1. click('Standard')
2. click('Black')
3. click('Buy Now')

Observation: Your score (min 0.0, max 1.0): 9.0

Thought: Purchase successful with score 9.0
Action: Finish(Purchase successful with score 9.0)
{END_OF_RESPONSE}
"""
]

def get_planner_prompt(fewshot: int):
    if fewshot > len(PLANNER_FEWSHOT_LIST):
        fewshot = len(PLANNER_FEWSHOT_LIST)
        print(f"Max fewshot examples for webshop planner prompt is {len(PLANNER_FEWSHOT_LIST)}. Running with {fewshot} fewshot examples.")
    return PLANNER_PROMPT + "\n".join(PLANNER_FEWSHOT_LIST[:fewshot])

def get_output_prompt(fewshot: int):
    if fewshot > len(OUTPUT_FEWSHOT_LIST):
        fewshot = len(OUTPUT_FEWSHOT_LIST)
        print(f"Max fewshot examples for webshop output prompt is {len(OUTPUT_FEWSHOT_LIST)}. Running with {fewshot} fewshot examples.")
    return OUTPUT_PROMPT + "\n".join(OUTPUT_FEWSHOT_LIST[:fewshot])