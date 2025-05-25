import requests
import json
import random
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_request_and_get_response(prompt: str) -> str:
    pre_prompt = """You are a math expert.
        You are given a math problem.
        You need to solve the math problem.
        You need to solve the math problem step by step.
        Your answer should be a number.
        Your answer should be between the following tag <answer> and </answer>
        Your answer should match this regex <answer>(-?\\d+)</answer>
        Here is an example question and answer
        question: "120 + 55"
        answer: <answer>175</answer>
        """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen3:0.6b", "prompt": f"{pre_prompt}\n{prompt}"},
    )

    tot = []
    for line in response.iter_lines():
        if line:
            line_json = json.loads(line.decode("utf-8"))
            if line_json["done"]:
                break
            tot.append(line_json["response"])

    to_ret = "".join(tot)
    logger.debug(f"LLM response:\n{to_ret}")
    return to_ret


def evaluate(prompt: str) -> bool:
    logger.debug(f"Evaluating: {prompt}")
    ans = eval(prompt)
    llm_res = send_request_and_get_response(prompt)

    match = re.search(r"<answer>(-?\d+)</answer>", llm_res)
    if not match:
        logger.error(f"Could not find answer in response: {llm_res}")
        return False
    llm_ans = int(match.group(1))

    to_ret = ans == llm_ans
    if not to_ret:
        logger.debug(f"{llm_res}\n\n{ans}")
    return to_ret


def construct_equation() -> str:
    num_low = 20
    num_high = 250
    nums_low = 2
    nums_high = 7
    to_ret = ""
    nums = random.randint(nums_low, nums_high)
    mult_used = 0
    for i in range(nums):
        to_ret += str(random.randint(num_low, num_high))
        if i != nums - 1:
            to_ret += " * " if mult_used < 2 else random.choice([" + ", " - ", " * "])
            mult_used += 1
    return to_ret


correct = 0
total = 100
for i in range(1, total + 1):
    if evaluate(construct_equation()):
        correct += 1
    if i % 5 == 0:
        logger.info(f"Progress: {i}/{total} correct: ({correct}/{i})")
print(f"Accuracy: {correct / total}")
