from lamoom import PipePrompt

agent = PipePrompt(id='lamoom_cicd__compare_results')
agent.add("""
You need to find answers on question in REAL_RESPONSE. Ask Question from QUESTIONS_AND_ANSWERS. QUESTIONS_AND_ANSWERS has ideal answer and questions;
For Each question there is an ideal answer and real_answer. You need to compare the ideal_answer and the real_answer.
If they match logically then this question matches, otherwise no.
If real_answer doesnt match ideal answer, say: "ANSWER_NOT_PROVIDED".
## IDEAL_ANSWER:
{ideal_answer}
""", role='system')

agent.add("""
# QUESTIONS_AND_ANSWERS
{generated_test}

# REAL_RESPONSE
{llm_response}

First, go through each question and get real_answer on it with from REAL_RESPONSE. Compare the answer you got from QUESTIONS_AND_ANSWERS. 
Secondly, check out if the real answer has other statements which are not in ideal answer.Add statements into JSON;
Finally, check out if the real answer has statements which contradicts ideal answer. Add statements into JSON;
Use the next JSON format for the answer:
# RESPONSE
```json
{
    "QUESTIONS_AND_ANSWERS": {
       "question from QUESTIONS_AND_ANSWERS": {
            "real_answer": "answer from the REAL_RESPONSE",
            "ideal_answer": "rewritten ideal answer from QUESTIONS_AND_ANSWERS",
            "does_match_with_ideal_answer": true/false
        },
        ...
        "last question from QUESTIONS_AND_ANSWERS": {
            "real_answer": "answer from the REAL_RESPONSE",
            "ideal_answer": "rewritten ideal answer from QUESTIONS_AND_ANSWERS",
            "does_match_with_ideal_answer": true/false
        }
    },
    "additional_statements_from_real_response": [ "statement1", "statement2", ...],
    "statements_which_contradict_ideal_answer_from_a_real_response": [ "statement1", "statement2", ...]
}
```
""", role='user')
