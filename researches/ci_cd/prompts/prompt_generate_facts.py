from lamoom import PipePrompt

agent = PipePrompt(id='lamoom_cicd__generate_facts')
agent.add("""
You're generating statements for the provided text below.
""", role='system')
agent.add("""
User asked:
{question}

# Ideal answer is:
{ideal_answer}
""", role='system')

agent.add("""
First, write out all the important statements from Ideal answer. 
    Make them data-enriched for each statement, so each statement is a fact. Where statement answers an questions, what, why, when, who, how and other questions if applicable from the Ideal answer or question;
    Tend to make a detailed statement. So it can be used out of the context of the Ideal answer and it will be true based on the Ideal answer.
Second, ask questions to each generated statement so that it produces a statement as an answer to the question.
Third, give a name to the generated facts, such as the name of the test, like "when_what_why_how_who_what_generated_name".
Use the next json format for the answer:
```json
{
    "statements": [
        "statement_from_ideal_answer",
        ...
        "another_statement_from_ideal_answer"
    ],
    "questions": {
        "statement_from_ideal_answer": "question_to_answer_by_that_statement",
        "another_statement_from_ideal_answer": "question_to_answer_by_that_another_statement",
        ...,
        "qN": "statementN",
    },
    "name": "when_what_why_generated_name"
}
```
""")

