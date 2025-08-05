from legal_chatbot_test.chatbot import get_chatbot_response
from evaluator import evaluate_response
from pathlib import Path
import json

TEST_FILES_DIR = Path(__file__).parent / "eval_questions"

def generate_test_input():
    if not TEST_FILES_DIR.exists():
        print(f"Test files directory does not exist: {TEST_FILES_DIR}")
        exit(1)

    test_inputs = []

    for file in TEST_FILES_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
            for test_case in test_data:
                question = test_case["question"]
                expected_answer = test_case["expected_answer"]
                chatbot_response = get_chatbot_response(question)
                test_inputs.append({
                    "question": question,
                    "response": chatbot_response,
                    "expected_answer": expected_answer
                })
    return test_inputs


def evaluate_test_cases(input):
    output = {
        "num_tests": 0,
        "num_passed": 0,
        "num_failed": 0,
        "results": []
    }
    for i, test_case in enumerate(input):
        output["num_tests"] += 1
        print(f"Evaluating test case {i+1}/{len(input)}: {test_case['question']}")
        evaluation_result = evaluate_response(test_case)
        result_dict = json.loads(evaluation_result.content)
        merged_dict = {**result_dict, **test_case}
        output["results"].append(merged_dict)
        if merged_dict["result"].lower() == "true":
            output["num_passed"] += 1
        else:
            output["num_failed"] += 1
    return output

if __name__ == "__main__":
    input = generate_test_input()
    output = evaluate_test_cases(input)
    print(json.dumps(output, indent=2))
    print("\n--- Test Summary ---")
    print(f"Total Tests: {output['num_tests']}")
    print(f"Passed: {output['num_passed']}")
    print(f"Failed: {output['num_failed']}")
    print("Test evaluation complete.")