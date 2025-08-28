from dataclasses import dataclass

@dataclass
class Statement:
    statement: str
    question: str
    
    def to_dict(self):
        return {
            "statement": self.statement,
            "question": self.question
        }

class Question:
    def __init__(self, test_question: str, llm_answer: str, ideal_answer: str, does_match_ideal_answer: bool):
        self.test_question = test_question
        self.llm_answer = llm_answer
        self.ideal_answer = ideal_answer
        self.does_match_ideal_answer = does_match_ideal_answer
        # Initialize score list with current match result
        self.score = [{"matches": does_match_ideal_answer, "llm_response": llm_answer}]

    def add_score(self, matches: bool, llm_response: str):
        """Add another score entry to the question's score history"""
        self.score.append({"matches": matches, "llm_response": llm_response})

    def to_dict(self):
        return {
            "test_question": self.test_question,
            "llm_answer": self.llm_answer,
            "ideal_answer": self.ideal_answer,
            "does_match_ideal_answer": self.does_match_ideal_answer,
            "score": self.score
        }
        
class Score:
    def __init__(self, score: int, passed: bool):
        self.score = score
        self.passed = passed
    
    def to_dict(self):
        return {
            "score": self.score,
            "passed": self.passed,
        }
        
@dataclass(kw_only=True)
class TestResult:
    prompt_id: str
    questions: list[Question]
    score: Score
    ideal_response: str
    llm_response: str
    statements: list[Statement] = None
    optional_params: dict = None
    statements_which_contradict: list[str] = None
    additional_statements: list[str] = None

    def to_dict(self):
        return {
            "prompt_id": self.prompt_id,
            "questions": [question.to_dict() for question in self.questions],
            "score": self.score.to_dict(),
            "ideal_response": self.ideal_response,
            "llm_response": self.llm_response,
            "statements": [statement.to_dict() for statement in self.statements] if self.statements else None,
            "optional_params": self.optional_params,
            "statements_which_contradict": self.statements_which_contradict,
            "additional_statements": self.additional_statements,
        }
