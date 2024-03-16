from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

#--------------------------------------------------------------------------------#
#                          Common variables
#--------------------------------------------------------------------------------#
dialect = "PostgreSQL"
top_k = "5"
table_description = "hoteldata: Remember the name of table in the database is called hoteldata. Contains table with information about hotel rooms. Useful to suggest hotels based on type of rooms, , location, ratings, amenities available etc"
table_schema = """CREATE TABLE hoteldata (
    room_type TEXT, 
    onsite_rate DOUBLE PRECISION, 
    room_amenities TEXT, 
    max_occupancy BIGINT, 
    roomdescription TEXT, 
    hotel_name TEXT, 
    city TEXT,
    country TEXT,
    star_rating BIGINT, 
    meals_included BOOLEAN
)"""

#--------------------------------------------------------------------------------#
#                          zero_shot_sql_prompt related variables
#--------------------------------------------------------------------------------#
zero_shot_sql_prompt_template = """\n\nHuman: You are an assistant with expertise to write syntactically correct SQL queries for {dialect} database given a human's question.

You have access to following table descriptions within <description></description> XML tags to help choose table(s) when answering:
<description>
{table_description}
</description>

You have access to following table schemas within <table></table> XML tags:
<table>
{table_schema}
</table>

You have access to below limit, only use when specifically not mentioned by human:
<limit>{top_k}</limit>

Make sure that you only choose tables from the <table> XML tags. Remember to respond with "NO_QUERY_TO_GENERATE" when the question does not correspond to any table description given.
ALWAYS use the following format:

Question: the input question
SQLQuery: the sql query to run

Assistant: Question: {question}
SQLQuery: """

zero_shot_sql_prompt = PromptTemplate(
    input_variables=[
        "question"
    ],
    partial_variables={
        "dialect": dialect,
        "top_k": top_k,
        "table_description": table_description,
        "table_schema": table_schema,
    },
    template=zero_shot_sql_prompt_template
)

#--------------------------------------------------------------------------------#
#                          few_shot_sql_prompt related variables
#--------------------------------------------------------------------------------#
prefix = """\n\nHuman: You are an assistant with expertise to write syntactically correct SQL queries for {dialect} database given a human's question.

You have access to following table descriptions within <description></description> XML tags to help choose table(s) when answering:
<description>
{table_description}
</description>

You have access to following table schemas within <table></table> XML tags:
<table>
{table_schema}
</table>

You have access to below limit, only use when specifically not mentioned by human:
<limit>{top_k}</limit>

<examples>"""

# ----------------------------------------- #
examples = [
    {
        "id": 1,
        "question": "I want a hotel in the Balkans with a king sized bed and a hot tub. Budget is $300 a night",
        "sql_query": "SELECT * FROM hoteldata WHERE onsite_rate <= 300 AND (country LIKE '%Bulgaria%' OR country LIKE '%Greece%' OR country LIKE '%Croatia%' OR country LIKE '%Serbia%') LIMIT 5"
    },
    {
        "id": 2,
        "question": "A room with breakfast included for 3 people, at a Hilton",
        "sql_query": "SELECT * FROM hoteldata WHERE meals_included = true AND max_occupancy >= 3 AND hotel_name LIKE '%Hilton%' LIMIT 5"
    },
    {
        "id": 3,
        "question": "Find a 2-person room in Vienna or London, preferably with meals included and AC",
        "sql_query": "SELECT * FROM hoteldata WHERE max_occupancy = 2 AND (city LIKE '%Vienna%' OR city LIKE '%London%') AND meals_included = true AND room_amenities LIKE '%AC%' LIMIT 5"
    }
]

example_template = """<example id={id}>
Question: {question}
SQLQuery: {sql_query}
</example>"""

example_prompt = PromptTemplate(
    input_variables=["id", "question", "sql_query"],
    template=example_template
)

# ----------------------------------------- #
suffix = """</examples>

Make sure that you only choose tables from the <table> XML tags. Remember to respond with "NO_QUERY_TO_GENERATE" when the question does not correspond to any table description given.
ALWAYS use the following format:

Question: the input question
SQLQuery: the sql query to run

Assistant: Question: {question}
SQLQuery: """

suffix_builtin = """</examples>

Make sure that you only choose tables from the <table> XML tags. Remember to respond with "NO_QUERY_TO_GENERATE" when the question does not correspond to any table description given.
ALWAYS use the following format:

Question: the input question
SQLQuery: the sql query to run

Assistant: Question: {input}
SQLQuery: """

suffix_with_history = """</examples>

Make sure that you only choose tables from the <table> XML tags. Remember to respond with "NO_QUERY_TO_GENERATE" when the question does not correspond to any table description given.
ALWAYS use the following format:

Question: the input question
SQLQuery: the sql query to run

You have access to previous conversation
<chat_history>
{chat_history}
</chat_history>

Assistant: Question: {question}
SQLQuery: """

# ----------------------------------------- #
few_shot_sql_prompt = FewShotPromptTemplate(
    input_variables=[
        "question"
    ],
    partial_variables={
        "dialect": dialect,
        "top_k": top_k,
        "table_description": table_description,
        "table_schema": table_schema,
    },
    prefix=prefix,
    examples=examples,
    example_prompt=example_prompt,
    example_separator="\n",  
    suffix=suffix,
)

few_shot_sql_builtin_prompt = FewShotPromptTemplate(
    input_variables=[
        "input"
    ],
    partial_variables={
        "dialect": dialect,
        "top_k": top_k,
        "table_description": table_description,
        "table_schema": table_schema,
    },
    prefix=prefix,
    examples=examples,
    example_prompt=example_prompt,
    example_separator="\n",  
    suffix=suffix_builtin,
)

few_shot_sql_prompt_with_history = FewShotPromptTemplate(
    input_variables=[
        "question",
        "chat_history"
    ],
    partial_variables={
        "dialect": dialect,
        "top_k": top_k,
        "table_description": table_description,
        "table_schema": table_schema,
    },
    prefix=prefix,
    examples=examples,
    example_prompt=example_prompt,
    example_separator="\n",  
    suffix=suffix_with_history,
)

#--------------------------------------------------------------------------------#
#                          few_shot_sql_improved_prompt related variables
#--------------------------------------------------------------------------------#
column_description = """Name: `room_type`
Description: 
Type: object
Distinct: ['Double Room', 'Family Room', 'Vacation Home', 'Triple Room', 'Suite', 'Twin Room', 'Quadruple Room', 'Superior Double Room', 'Junior Suite', 'Double or Twin Room', 'Deluxe Double Room', 'Standard Double Room', 'Single Room', 'Family Room (2 Adults + 2 Children)', 'Apartment', 'Triple', 'Comfort Double Room', 'Standard Twin Room', 'Family Suite', 'Single']

Name: `onsite_rate`
Description: 
Type: float64
Distinct: [0.0, 95.03, 126.71, 89.75, 90.42, 147.83, 84.48, 100.32, 79.19, 80.37, 116.16, 110.51, 137.27, 158.39, 131.99, 105.59, 104.54, 142.55, 85.39, 75.35]

Name: `room_amenities`
Description: 
Type: object
Distinct: [Additional bathroom', 'Additional toilet', 'Air conditioning', 'Air purifier', 'Alarm clock', 'Bathrobes', 'Bathroom phone', 'Blackout curtains', 'Carbon monoxide detector', 'Carpeting', 'Cleaning products', 'Closet', 'Clothes dryer', 'Clothes rack', 'Coffee/tea maker', 'Complimentary tea', 'DVD/CD player', 'Daily housekeeping', 'Daily newspaper', 'Dart board']

Name: `max_occupancy`
Description: 
Type: int64
Distinct: [2, 1, 4, 3, 6, 5, 8, 10, 7, 9, 12, 13, 15, 14, 17, 24, 16, 20, 11]

Name: `roomdescription`
Description: 
Type: object
Distinct: [1 bunk bed', '1 double bed', '1 futon', '1 king bed', '1 queen bed', '1 semi double bed', '1 single bed', '1 sofa bed', '1 super king bed', '10 bathrooms', '10 bunk beds', '10 queen beds', '12 double beds', '17 king beds', '2 bathrooms', '2 bedrooms', '2 bunk beds', '2 double beds', '2 futons', '2 king beds']

Name: `hotel_name`
Description: 
Type: object
Distinct: ['Hotel Europa', 'The Royal Hotel', 'Castle Hotel', 'Hotel Panorama', 'North Stafford Hotel Town Centre', 'Hotel Eden', 'The Castle Hotel', 'Holiday Inn Express Manchester City Centre Arena', 'Swan Hotel by Greene King Inns', 'The Red Lion', 'The Ship Inn', 'The Park Hotel', 'Sporthotel Igls', 'Hotel Post', 'Kreutzwald Hotel Tallinn', 'Hotel Universal', 'Best Western Plus Windmill Village Hotel', 'The Kings Arms Hotel', 'Travelodge Nottingham Central', 'Hotel Dona Mayor']

Name: `city`
Description: 
Type: object
Distinct: ['London', 'Paris', 'Rome', 'Manchester', 'Edinburgh', 'Berlin', 'Prague', 'Barcelona', 'Munich', 'Birmingham', 'Madrid', 'Athens', 'Glasgow', 'Milan', 'Amsterdam', 'Aberdeen', 'Florence', 'Venice', 'Liverpool', 'Bournemouth']

Name: `country`
Description: 
Type: object
Distinct: ['United Kingdom', 'France', 'Italy', 'Germany', 'Spain', 'Greece', 'Poland', 'Switzerland', 'Austria', 'Czech Republic', 'Netherlands', 'Portugal', 'Romania', 'Belgium', 'Bulgaria', 'Hungary', 'Sweden', 'Ireland', 'Denmark', 'Slovakia']

Name: `star_rating`
Description: 
Type: int64
Distinct: [3, 4, 2]

Name: `meals_included`
Description: 
Type: bool
Distinct: [True, False]"""

prefix_improved = """\n\nHuman: You are an assistant with expertise to write syntactically correct SQL queries for {dialect} database given a human's question.

You have access to following table descriptions within <description></description> XML tags to help choose table(s) when answering:
<description>
{table_description}
</description>

You have access to following table schemas within <table></table> XML tags:
<table>
{table_schema}
</table>

You have access to following descriptions of columns of a table <column_description> XML tags:
<column_description>
{column_description}
</column_description>

You have access to below limit, only use when specifically not mentioned by human:
<limit>{top_k}</limit>

<examples>"""

table_schema_improved = """CREATE TABLE hoteldata (
    room_type TEXT, 
    onsite_rate DOUBLE PRECISION, 
    room_amenities TEXT, 
    max_occupancy BIGINT, 
    roomdescription TEXT, 
    hotel_name TEXT, 
    city TEXT, NOTE: Always use LIKE operator with this column
    country TEXT, NOTE: If a region is mentioned, include all relevant countries.
    star_rating BIGINT, 
    meals_included BOOLEAN
)"""

few_shot_sql_improved_prompt = FewShotPromptTemplate(
    input_variables=[
        "question"
    ],
    partial_variables={
        "dialect": dialect,
        "top_k": top_k,
        "table_description": table_description,
        "table_schema": table_schema_improved,
        "column_description": column_description,
    },
    prefix=prefix_improved,
    examples=examples,
    example_prompt=example_prompt,
    example_separator="\n",  
    suffix=suffix,
)

#--------------------------------------------------------------------------------#
#                          sql_to_natural_lang related variables
#--------------------------------------------------------------------------------#

text_generate_prompt_template = """\n\nHuman: You are an expert in summarizing SQL responses. 
You have access to table schema in <table></table> XML tags:
<table>
{table_schema}
</table>

Skip the preamble and write less than 2 lines summary of SQL Response, based on the table schema, Question, SQL Query, and SQL Response:

Question: {question}
SQL Query: {sql_query}
SQL Response: {db_response}

Assistant: Summary: """

text_generate_prompt = PromptTemplate(
    input_variables=[
        "question",
        "sql_query",
        "db_response"
    ],
    partial_variables={
        "table_schema": table_schema,
    },
    template=text_generate_prompt_template 
)

text_generate_prompt_template_with_history = """\n\nHuman: You are an expert in summarizing SQL responses. 
You have access to table schema in <table></table> XML tags:
<table>
{table_schema}
</table>

You have access to previous conversation:
<chat_history>
{chat_history}
</chat_history>

Skip the preamble and write less than 2 lines summary of SQL Response, based on the table schema, Question, SQL Query, and SQL Response:

Question: {question}
SQL Query: {sql_query}
SQL Response: {db_response}

Assistant: Summary: """

text_generate_prompt_with_history = PromptTemplate(
    input_variables=[
        "chat_history",
        "question",
        "sql_query",
        "db_response"
    ],
    partial_variables={
        "table_schema": table_schema,
    },
    template=text_generate_prompt_template_with_history 
)
#--------------------------------------------------------------------------------#
#                          agent related variables
#--------------------------------------------------------------------------------#

agent_prompt_template = """\n\nHuman: The following is a conversation between a human and an AI assistant. The assistant is polite, and responds to the user input and questions acurately and concisely. The assistant stays on the topic of the user input and does not diverge from it. You will play the role of the assistant.

You have access to the following tools:
{tools}

Use the following format:

The $JSON_BLOB should only contain a SINGLE action and MUST be formatted as markdown, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time from in this format:
Action:
```
$JSON_BLOB
```
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Make sure that Action should be one of [{tool_names}].  
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```
Remember to respond with your knowledge when the question does not correspond to any tool.

Begin!
Question: {question}

Assistant:
{agent_scratchpad}
"""

agent_prompt_template_with_history = """\n\nHuman: The following is a conversation between a human and an AI assistant. The assistant is polite, and responds to the user input and questions acurately and concisely. The assistant stays on the topic of the user input and does not diverge from it. You will play the role of the assistant.

You have access to the following tools:
{tools}

Use the following format:

The $JSON_BLOB should only contain a SINGLE action and MUST be formatted as markdown, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time from in this format:
Action:
```
$JSON_BLOB
```
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Make sure that Action should be one of [{tool_names}].  
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```
Remember to respond with your knowledge when the question does not correspond to any tool.

Begin!

You have access to previous conversation:
<chat_history>
{chat_history}
</chat_history>

Question: {question}

Assistant:
{agent_scratchpad}
"""