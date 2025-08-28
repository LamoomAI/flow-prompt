import pandas as pd
import json
import logging
import csv
import os
from typing import List, Dict, Any, Optional
from responses import Question, TestResult, Score, Statement

logger = logging.getLogger(__name__)

def parse_csv_file(file_path: str) -> list:
    """
    Reads a CSV file and returns a list of dictionaries with keys:
    - ideal_answer
    - llm_response
    - optional_params (parsed as dict if not empty)
    - generated_test (parsed as dict if present)
    
    This function is compatible with CSV files created by export_generated_tests_to_csv.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []

    test_cases = []
    for _, row in df.iterrows():
        case = {
            "ideal_answer": row.get("ideal_answer"),
            "llm_response": row.get("llm_response")
        }
        
        # Parse optional_params
        opt_params = row.get("optional_params")
        if pd.notna(opt_params) and opt_params:
            try:
                case["optional_params"] = json.loads(opt_params)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing optional_params: {e}")
                case["optional_params"] = None
        else:
            case["optional_params"] = None
            
        # Parse generated_test if present
        generated_test = row.get("generated_test")
        if pd.notna(generated_test) and generated_test:
            try:
                parsed_test = json.loads(generated_test)
                case["generated_test"] = parsed_test
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing generated_test: {e}")
                case["generated_test"] = None
        
        test_cases.append(case)

    return test_cases


def export_generated_tests_to_csv(file_path: str, tests: List[dict]):
    """
    Exports generated tests to a CSV file.
    
    Args:
        file_path: Path to save the CSV file
        tests: List of test dictionaries containing ideal_answer, llm_response, etc.
    """
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['ideal_answer', 'llm_response', 'optional_params', 'generated_test'])
            
            # Write rows
            for test in tests:
                ideal_answer = test.get('ideal_answer', '')
                llm_response = test.get('llm_response', '')
                optional_params = json.dumps(test.get('optional_params', {})) if test.get('optional_params') else ''
                
                # Convert statements and questions to JSON string
                generated_test = {}
                if 'statements' in test and test['statements']:
                    statements_list = []
                    questions_dict = {}
                    
                    for statement in test['statements']:
                        statement_text = statement.get('statement', '')
                        question_text = statement.get('question', '')
                        statements_list.append(statement_text)
                        questions_dict[statement_text] = question_text
                    
                    generated_test = {
                        'statements': statements_list,
                        'questions': questions_dict
                    }
                
                generated_test_json = json.dumps(generated_test)
                
                writer.writerow([ideal_answer, llm_response, optional_params, generated_test_json])
                
        logger.info(f"Generated tests exported to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting generated tests to CSV: {e}")
        return False


def export_results_to_csv(file_path: str, results: List):
    """
    Exports test results to a CSV file with detailed information.
    
    Args:
        file_path: Path to save the CSV file
        results: List of TestResult objects
    """
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header for detailed results
            writer.writerow([
                'prompt_id', 
                'prompt_version',
                'question', 
                'statement', 
                'llm_response_to_question', 
                'ideal_answer', 
                'does_match_ideal_answer',
                'score',
                'full_llm_response', 
                'full_ideal_answer'
            ])
            
            # Write rows for each question in each result
            for result in results:
                prompt_id = result.prompt_id
                
                # Extract prompt_version from optional_params if available
                prompt_version = ''
                if result.optional_params and 'prompt_version' in result.optional_params:
                    prompt_version = result.optional_params['prompt_version']
                
                # Create a mapping of questions to statements
                question_to_statement = {}
                if result.statements:
                    for statement in result.statements:
                        question_to_statement[statement.question] = statement.statement
                
                # Write a row for each question
                for question in result.questions:
                    test_question = question.test_question
                    statement = question_to_statement.get(test_question, '')
                    llm_answer = question.llm_answer
                    ideal_answer = question.ideal_answer
                    does_match = 'Yes' if question.does_match_ideal_answer else 'No'
                    
                    # Get score history as string
                    score_history = json.dumps(question.score)
                    
                    # Write the row
                    writer.writerow([
                        prompt_id,
                        prompt_version,
                        test_question,
                        statement,
                        llm_answer,
                        ideal_answer,
                        does_match,
                        score_history,
                        result.llm_response,
                        result.ideal_response
                    ])
                
        logger.info(f"Test results exported to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting results to CSV: {e}")
        return False

def import_results_from_csv(file_path: str) -> List[TestResult]:
    """
    Imports test results from a CSV file.
    
    Args:
        file_path: Path to the CSV file with results
        
    Returns:
        List of TestResult objects reconstructed from the CSV
    """
    try:
        df = pd.read_csv(file_path)
        
        # Group by prompt_id, full_llm_response, and full_ideal_answer
        # This helps us recreate TestResult objects
        grouped_data = df.groupby(['prompt_id', 'full_llm_response', 'full_ideal_answer'])
        
        results = []
        
        for (prompt_id, llm_response, ideal_response), group in grouped_data:
            # Extract prompt_version from the first row in the group
            prompt_version = group['prompt_version'].iloc[0] if 'prompt_version' in group.columns else None
            
            # Create optional_params dict if prompt_version exists
            optional_params = {'prompt_version': prompt_version} if pd.notna(prompt_version) and prompt_version else {}
            
            # Create Question objects
            questions = []
            statements = []
            
            for _, row in group.iterrows():
                # Create Question object
                test_question = row['question']
                llm_answer = row['llm_response_to_question']
                ideal_answer = row['ideal_answer']
                does_match = True if row['does_match_ideal_answer'] == 'Yes' else False
                
                q = Question(test_question, llm_answer, ideal_answer, does_match)
                
                # If score history is available, add it
                if 'score' in row and pd.notna(row['score']):
                    try:
                        score_history = json.loads(row['score'])
                        q.score = score_history
                    except json.JSONDecodeError:
                        pass
                
                questions.append(q)
                
                # Create Statement object if statement column is available
                if 'statement' in row and pd.notna(row['statement']) and row['statement']:
                    statement = row['statement']
                    statements.append(Statement(statement=statement, question=test_question))
            
            # Calculate the score
            passed_count = sum(1 for q in questions if q.does_match_ideal_answer)
            score_value = round(passed_count / len(questions) * 100) if questions else 0
            passed = score_value >= 70  # Using default threshold of 70%
            score = Score(score_value, passed)
            
            # Create TestResult object
            test_result = TestResult(
                prompt_id=prompt_id,
                questions=questions,
                score=score,
                ideal_response=ideal_response,
                llm_response=llm_response,
                statements=statements if statements else None,
                optional_params=optional_params if optional_params else None
            )
            
            results.append(test_result)
        
        logger.info(f"Imported {len(results)} test results from {file_path}")
        return results
    except Exception as e:
        logger.error(f"Error importing results from CSV: {e}")
        return []