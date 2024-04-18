import json
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains.query_constructor.base import get_query_constructor_prompt

# ------------------------------------------------------------------------------------------------ #
#                                 Common variables                                                 #
# ------------------------------------------------------------------------------------------------ #
document_contents = "Detailed description of hotel rooms"
attribute_info = [
    {
        "name": "onsite_rate",
        "description": "The average daily rate for a room at this hotel",
        "type": "float"
    },
    {
        "name": "max_occupancy",
        "description": "The maximum number of guests allowed per room at this hotel",
        "type": "integer"
    },
    {
        "name": "city",
        "description": "The city where this hotel is located",
        "type": "string"
    },
    {
        "name": "country",
        "description": "The country where this hotel is located",
        "type": "string"
    },
    {
        "name": "star_rating",
        "description": "The star rating for this hotel, on a scale of 1 to 5 stars",
        "type": "integer"
    },
    {
        "name": "meals_included",
        "description": "Whether meals are included in the room rate at this hotel",
        "type": "boolean"
    }
]
input_output_pairs = [
    (
        "All hotels with no wifi",
        {
            "query": "",
            "filter": 'NO_FILTER',
        },
    ),
    (
        "I want a hotel in the Balkans with a king sized bed and a hot tub. Budget is $300 a night",
        {
            "query": "king-sized bed, hot tub",
            "filter": 'and(in("country", ["Bulgaria", "Greece", "Croatia", "Serbia"]), lte("onsiterate", 300))',
        },
    ),
    (
        "A room with breakfast included for 3 people, at a Hilton",
        {
            "query": "Hilton",
            "filter": 'and(eq("mealsincluded", true), gte("maxoccupancy", 3))',
        },
    ),

]

# ------------------------------------------------------------------------------------------------ #
#                                  Default prompt related variables                                #
# ------------------------------------------------------------------------------------------------ #
default_prompt = get_query_constructor_prompt(
    document_contents, attribute_info, examples=input_output_pairs
)
# print(default_prompt.format(query="{query}"))


# ------------------------------------------------------------------------------------------------ #
#                                   Few Shot Prompt related variables                              #
# ------------------------------------------------------------------------------------------------ #
def format_attribute_info(info):
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=4).replace("{", "{{").replace("}", "}}")

def construct_examples(input_output_pairs):
    examples = []
    for i, (_input, output) in enumerate(input_output_pairs):
        structured_request = (
            json.dumps(output, indent=4).replace("{", "{{").replace("}", "}}")
        )
        example = {
            "i": i + 1,
            "user_query": _input,
            "structured_request": structured_request,
        }
        examples.append(example)
    return examples

# allowed_comparators = [
#     Comparator.EQ, Comparator.CONTAIN, Comparator.LIKE,
#     Comparator.LT, Comparator.LTE,
#     Comparator.GT, Comparator.GTE,
# ]

# allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]

# allowed_attributes = [attr["name"] for attr in attribute_info]

schema = """<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{{{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
    "limit": int \ the number of documents to retrieve
}}}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
- `attr` (string):  name of attribute to apply the comparison to, put always in double quotes
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

<rules>
1. You are only allowed to use these operators: 
["eq", "ne", "gt", "gte", "lt", "lte", "contain", "like", "in", "nin", "and", "or", "not"]
Do not use ["limit", "lower"] as operators.
Make sure the operators are always wrapped within double quotes
2. Make sure that you only use the comparators and logical operators listed above and no others.
3. Make sure that filters only refer to attributes that exist in the data source.
4. Make sure that filters only use the attributed names with its function names if there are functions applied on them.
5. Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
6. Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
7. Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
</rules>"""

prefix_template = """\n\nHuman: Your goal is to structure the user's query to match the request schema provided below.

<schema>
{schema}
</schema>

Use the below data source to answer Questions.
<data_source>
```json
{{{{
    "content": "{content}",
    "attributes": {attributes}
}}}}
```
</data_source>"""

suffix_template = """Question: {query}
Assistant: 
Structured Request: """

example_template = """<example id={i}>
Question: {user_query}
Structured Request:
```json
{structured_request}
```
</example>"""


attribute_str = format_attribute_info(attribute_info)

examples = construct_examples(input_output_pairs)

example_prompt = PromptTemplate(
    input_variables=["i", "user_query", "structured_request"],
    template=example_template,
)

prefix = prefix_template.format(
    schema=schema, content=document_contents, attributes=attribute_str
)

suffix = suffix_template

few_shot_self_query_prompt = FewShotPromptTemplate(
    examples=list(examples),
    example_prompt=example_prompt,
    input_variables=["query"],
    suffix=suffix,
    prefix=prefix,
)

