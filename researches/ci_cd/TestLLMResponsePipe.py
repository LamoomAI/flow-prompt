import json
import logging
import numpy as np
from lamoom import Lamoom
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from prompts.prompt_generate_facts import agent as generate_facts_agent
from prompts.prompt_compare_results import agent as compare_results_agent

from responses import Question, TestResult, Score, Statement
from exceptions import GenerateFactsException

from utils import parse_csv_file, export_generated_tests_to_csv, export_results_to_csv, import_results_from_csv

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TestLLMResponsePipe:
    lamoom: Lamoom = None
    
    threshold: int = 70
    
    accumulated_results: list[TestResult] = field(default_factory=list)

    @classmethod
    def generate_statements(cls, question: str, ideal_answer: str) -> List[Statement]:
        """
        Generates statements and corresponding questions from the ideal answer.
        
        Args:
            ideal_answer: The ideal answer to extract statements and generate questions from
            
        Returns:
            A list of Statement objects containing statement and question pairs
        """
        # Call the LLM to generate facts/statements and questions
        response = self.lamoom.call(generate_facts_agent.id, {"question": question, "ideal_answer": ideal_answer}, "gemini/gemini-2.5-pro")
        
        # Parse the response
        print( response.content )
        result = response.parsed_json
        statements_list = result.get("statements", [])
        questions_dict = result.get("questions", {})
        
        # Create Statement objects
        statement_objects = []
        
        # First gather statements with explicit questions
        for statement, question in questions_dict.items():
            statement_objects.append(Statement(statement=statement, question=question))
        
        # Check if we have any standalone statements without questions
        # This is a fallback in case the LLM doesn't pair all statements with questions
        for statement in statements_list:
            # Check if this statement is already included
            if not any(s.statement == statement for s in statement_objects):
                # Create a default question for this statement
                default_question = f"Does the response mention that {statement}?"
                statement_objects.append(Statement(statement=statement, question=default_question))
        
        if not statement_objects:
            raise GenerateFactsException("No statements or questions were generated.")
            
        return statement_objects
    
    def get_generated_test(self, statements: List[Statement]):
        """
        Converts a list of Statement objects to the format needed for comparison.
        """
        generated_test = {}
        for statement in statements:
            generated_test[statement.question] = {
                'answer_by_llm': statement.statement
            }
        
        if len(generated_test.items()) == 0:
            raise GenerateFactsException("No questions were generated.")
        
        return generated_test
    
    def calculate_score(self, test_results: dict, threshold: int) -> Score:
        pass_count = 0
        question_numb = len(test_results.items()) or 1
        for _, values in test_results.items():
            if values['does_match_with_ideal_answer']:
                pass_count += 1
            
        score = round(pass_count / question_numb * 100)
        passed = True if score >= threshold else False
        
        return Score(score, passed)
    
    def compare(self, ideal_answer: str, 
                llm_response: str, 
                optional_params: dict = None,
                statements: List[Statement] = None) -> TestResult:
        """
        Compare an ideal answer with an LLM response.
        
        Args:
            ideal_answer: The ideal answer to compare against
            llm_response: The LLM-generated response to evaluate
            optional_params: Optional parameters like prompt_id
            input_generated_tests: Optional pre-generated statement-question pairs
                                  (if provided, skips generating them from the ideal answer)
        
        Returns:
            TestResult object containing the comparison results
        """
        # Use provided statements or generate them from the ideal answer
        if not statements:
            statements = self.generate_statements(ideal_answer)
        
        # Convert statements to the format needed for comparison
        generated_test = json.dumps(self.get_generated_test(statements))
        prompt_id = "user_prompt"
        if optional_params is not None:
            logger.info(optional_params)
            prompt_id = optional_params.get('prompt_id', "user_prompt")
            # TODO: Service CI/CD Logic

        # Compare results
        comparison_context = {
            "ideal_answer": ideal_answer,
            "generated_test": generated_test,
            "llm_response": llm_response,
        }
        comparison_response = self.lamoom.call(compare_results_agent.id, comparison_context, "gemini/gemini-2.5-pro")
        test_results = comparison_response.parsed_json.get("QUESTIONS_AND_ANSWERS", {})

        # Format results into Question objects
        questions_list = [
            Question(q, v["real_answer"], v["ideal_answer"], v["does_match_with_ideal_answer"])
            for q, v in test_results.items()
        ]
        
        score = self.calculate_score(test_results, self.threshold)
        
        test_result = TestResult(
            prompt_id=prompt_id,
            questions=questions_list,
            statements_which_contradict=test_results.get("statements_which_contradict_ideal_answer_from_a_real_response", []),
            additional_statements=test_results.get("additional_statements_from_real_response", []),
            score=score,
            ideal_response=ideal_answer,
            llm_response=llm_response,
            statements=statements,
            optional_params=optional_params
        )
        self.accumulated_results.append(test_result)

        return test_result
    
    
    def compare_from_csv(self, csv_file: str) -> list[TestResult]:
        """
        Reads a CSV file and runs compare() for each row.
        
        Expected columns:
        - ideal_answer (required): The ideal answer text
        - llm_response (required): The LLM's response text
        - optional_params (optional): JSON string with parameters like prompt_id
        - generated_test (optional): JSON string with pre-generated statements and questions
        
        This method is compatible with CSV files created by export_generated_tests_to_csv.
        
        Returns:
            List of TestResult objects
        """
        test_cases = parse_csv_file(csv_file)
        results = []
        for i, row in enumerate(test_cases):
            ideal_answer = row.get("ideal_answer")
            llm_response = row.get("llm_response")
            optional_params = row.get("optional_params")
            
            # Check if pre-generated tests are provided in the CSV
            input_generated_tests = []
            generated_test_data = row.get("generated_test")\

            if generated_test_data:
                # Convert the generated_test data to Statement objects
                for pair in generated_test_data:
                    question_text = pair.get("question")
                    statement_text = pair.get("statement")
                    input_generated_tests.append(Statement(statement=statement_text, question=question_text))
            if input_generated_tests is None:
                logger.info(f"Generating statements for test case {i + 1}")
                input_generated_tests = self.generate_statements(ideal_answer)
            
            logger.info(f"Comparing test case {i + 1}")
            test_result = self.compare(
                ideal_answer, 
                llm_response, 
                optional_params,
                input_generated_tests
            )
            logger.info(f"Test case {i + 1} score: {test_result.score.score}%")
            
            # Don't add it again since compare() already does it
            results.append(test_result)
        
        return results
    
    def import_results(self, file_path: str, label: str = None, update_optional_params: Dict = None):
        """
        Import results from a previously exported CSV file.
        
        Args:
            file_path: Path to the CSV file with results
            label: Optional label to distinguish imported results
            update_optional_params: Optional dictionary to update optional_params of imported results
                                   (useful for tagging imported results with version info)
        
        Returns:
            List of TestResult objects that were imported
        """
        imported_results = import_results_from_csv(file_path)
        
        # Update optional_params if provided
        if update_optional_params:
            for result in imported_results:
                if not result.optional_params:
                    result.optional_params = {}
                result.optional_params.update(update_optional_params)
        
        # Add label to prompt_id if provided
        if label:
            for result in imported_results:
                result.prompt_id = f"{result.prompt_id}_{label}"
        
        # Add imported results to accumulated results
        self.accumulated_results.extend(imported_results)
        
        logger.info(f"Imported {len(imported_results)} results from {file_path}")
        return imported_results
    
    def get_statistics(self, by_prompt_id: bool = True, by_ideal_answer: bool = False):
        """
        Calculate statistics for accumulated results.
        
        Args:
            by_prompt_id: Group statistics by prompt_id
            by_ideal_answer: Group statistics by ideal_answer
            
        Returns:
            Dictionary with statistics grouped according to parameters
        """
        if not self.accumulated_results:
            return {"error": "No results available"}
        
        stats = {}
        
        if by_prompt_id:
            prompt_stats = defaultdict(list)
            for result in self.accumulated_results:
                prompt_id = result.prompt_id
                score = result.score.score
                prompt_stats[prompt_id].append(score)
            
            for prompt_id, scores in prompt_stats.items():
                stats[prompt_id] = {
                    "count": len(scores),
                    "mean": np.mean(scores),
                    "median": np.median(scores),
                    "std": np.std(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "pass_rate": sum(1 for s in scores if s >= self.threshold) / len(scores)
                }
    
        # Create a hash of the ideal answer to use as a grouping key
        ideal_stats = defaultdict(list)
        ideal_to_id = {}  # Mapping from hash to actual answer text
        
        for result in self.accumulated_results:
            # Use first 50 chars as a readable ID
            ideal_id = result.ideal_response[:50] + "..." if len(result.ideal_response) > 50 else result.ideal_response
            ideal_to_id[ideal_id] = result.ideal_response
            ideal_stats[ideal_id].append(result.score.score)
        
        for ideal_id, scores in ideal_stats.items():
            stats[f"ideal_{ideal_id}"] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores),
                "pass_rate": sum(1 for s in scores if s >= self.threshold) / len(scores)
            }
        
        return stats
    
    def visualize_test_results(self, group_by: str = None,
                               show_statistics: bool = False, save_path: Optional[str] = None,
                               plt_figure_size: tuple = (12, 8)):
        """
        Plots visualization of accumulated scores based on specified parameters.
        
        Args:
            group_by: Group visualizations by value for x-axis ('optional_param__prompt_id' or 'ideal_answer', or number)
            show_statistics: Whether to show statistics (mean, median, std)
            save_path: Optional path to save the visualization
        """
        if not self.accumulated_results:
            logger.warning("No results to visualize")
            return
        
        # Create figure
        plt.figure(figsize=plt_figure_size)
        

        # Group scores by prompt_id
        groups = defaultdict(list)
        group_names = {}
        if not group_by:
            group_by = "prompt_id"
        for item in self.accumulated_results:
            if group_by.startswith("optional_param__"):
                group_by_value = hash(item.optional_params.get(group_by.split("__")[1], "default"))
                group_by_name = item.optional_params.get(group_by.split("__")[1], "default")
            else:
                group_by_value = hash(item.to_dict().get(group_by, "default"))
                group_by_name = item.to_dict().get(group_by, "default")
            score = item.score.score
            groups[group_by_value].append(score)
            group_names[group_by_value] = group_by_name
            
            # Plot scores by prompt_id
            max_length = 0
            for prompt_id, scores in groups.items():
                x_values = list(range(1, len(scores) + 1))
                plt.plot(x_values, scores, marker='o', linestyle='-')
                max_length = max(max_length, len(scores))
                
                # Add mean line if show_statistics is True
                if show_statistics and len(scores) > 1:
                    mean_score = np.mean(scores)
                    plt.axhline(y=mean_score, color=plt.gca().lines[-1].get_color(), 
                              linestyle='--', alpha=0.5)
            
            plt.title(f"LLM Test Scores per Prompt (Passing score = {self.threshold}%, Average={np.mean([s for sublist in groups.values() for s in sublist]):.2f}%)")
            plt.xlabel("Test Instance")
            plt.xticks(range(1, max_length + 1))
            plt.ylabel("Score (%)")
            plt.legend()
            plt.grid(True)
            
        # Add horizontal line for threshold
        plt.axhline(y=self.threshold, color='r', linestyle='-', label=f"Threshold ({self.threshold}%)")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
            
        plt.show()
        
    def visualize_statistics(self, by_prompt_id: bool = True, by_ideal_answer: bool = False,
                           visualization_type: str = "bar", save_path: Optional[str] = None):
        """
        Visualize statistics for accumulated results.
        
        Args:
            by_prompt_id: Include statistics grouped by prompt_id
            by_ideal_answer: Include statistics grouped by ideal_answer
            visualization_type: Type of visualization ('bar', 'box', or 'violin')
            save_path: Optional path to save the visualization
        """
        if not self.accumulated_results:
            logger.warning("No results to visualize statistics")
            return
        
        # Get statistics
        stats = self.get_statistics(by_prompt_id=by_prompt_id, by_ideal_answer=by_ideal_answer)
        
        if not stats:
            logger.warning("No statistics available")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        if visualization_type == "bar":
            # Extract data for bar chart
            groups = list(stats.keys())
            means = [stats[g]["mean"] for g in groups]
            stds = [stats[g]["std"] for g in groups]
            
            # Create bar chart with error bars
            plt.bar(groups, means, yerr=stds, capsize=5, alpha=0.7)
            plt.axhline(y=self.threshold, color='r', linestyle='-', label=f"Threshold ({self.threshold}%)")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Mean Score (%)")
            plt.title("Mean Scores with Standard Deviation")
            plt.legend()
            
        elif visualization_type == "box" or visualization_type == "violin":
            # Prepare data for box or violin plot
            data = []
            labels = []
            
            for group, stat in stats.items():
                # For each result in this group, add its score to the data
                group_scores = []
                for result in self.accumulated_results:
                    if (by_prompt_id and result.prompt_id == group) or \
                       (by_ideal_answer and group.startswith("ideal_") and result.ideal_response[:50] in group):
                        group_scores.append(result.score.score)
                
                if group_scores:
                    data.append(group_scores)
                    labels.append(group)
            
            if visualization_type == "box":
                # Create box plot
                plt.boxplot(data, labels=labels)
                plt.axhline(y=self.threshold, color='r', linestyle='-', label=f"Threshold ({self.threshold}%)")
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Score (%)")
                plt.title("Distribution of Scores")
            
            else:  # violin plot
                # Create violin plot
                plt.violinplot(data, showmeans=True, showmedians=True)
                plt.axhline(y=self.threshold, color='r', linestyle='-', label=f"Threshold ({self.threshold}%)")
                plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                plt.ylabel("Score (%)")
                plt.title("Distribution of Scores")
            
            plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Statistics visualization saved to {save_path}")
            
        plt.show()
        
    def export_generated_tests(self, file_path: str) -> bool:
        """
        Export generated tests with statements and questions to a CSV file.
        
        Args:
            file_path: Path to save the CSV file
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        test_data = []
        
        for result in self.accumulated_results:
            test_info = {
                'ideal_answer': result.ideal_response,
                'llm_response': result.llm_response,
                'optional_params': result.optional_params,
                'statements': [statement.to_dict() for statement in result.statements] if result.statements else []
            }
            test_data.append(test_info)
            
        return export_generated_tests_to_csv(file_path, test_data)
    
    def export_results(self, file_path: str) -> bool:
        """
        Export detailed test results to a CSV file.
        
        Args:
            file_path: Path to save the CSV file
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        return export_results_to_csv(file_path, self.accumulated_results)
