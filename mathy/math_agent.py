import requests
import json
import random
import logging
import re
import operator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# constants
MODEL = "gemma3:4b"
SIMPLE_PROMPT = """
You are a math expert.

Solve the following math problem. Only return the final answer.

Wrap your answer in <answer>...</answer> tags.

Your answer must match this format: <answer>(-?\d+)</answer>

Example:
Question: 120 + 55 * 2
Answer: <answer>230</answer>
"""
AGENT_PROMPT = """
You are a math expert.

You will be given a simple arithmetic equation (e.g., "5 + 2 * 10"). Your task is to solve it step by step using only the tools provided.

⚒️ Available tools:
<tool>add(x,y)</tool>         – Computes x + y  
<tool>subtract(x,y)</tool>    – Computes x - y  
<tool>multiply(x,y)</tool>    – Computes x * y  

🔧 Instructions:
1. Break down the equation and follow correct order of operations (PEMDAS).
2. Emit **one** <tool>(...) call at a time.
3. Wait for the result to be returned in a <tool-result>...</tool-result> tag.
4. Use that result in the next step, if needed.
5. When the final result is known, output it as: <answer>...</answer>
6. Your final answer must match this pattern: <answer>(-?\d+)</answer>

📘 Example:

Question: "5 + 2 * 10"

Step 1:
<tool>multiply(2,10)</tool>

User replies:
<tool-result>20</tool-result>

Step 2:
<tool>add(5,20)</tool>

User replies:
<tool-result>25</tool-result>

Final output:
<answer>25</answer>
"""


def send_request_and_get_response(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": f"{prompt}"},
    )

    tot = []
    for line in response.iter_lines():
        if line:
            line_json = json.loads(line.decode("utf-8"))
            if line_json["done"]:
                break
            tot.append(line_json["response"])

    to_ret = "".join(tot)
    return to_ret


def evaluate(pre_prompt: str, prompt: str) -> bool:
    ans = eval(prompt)
    logger.debug(f"Evaluating: {prompt} with answer: {ans}")

    conversation = [
        f"SYSTEM PROMPT: {pre_prompt}",
        f'Here is the equation you need to solve: "{prompt}"',
    ]
    num_iter = 0
    while num_iter < 3:
        num_iter += 1
        final_prompt = "\n".join(conversation)
        llm_res = send_request_and_get_response(final_prompt)
        conversation.append(llm_res)

        logger.debug(
            f"{'=' * 20}\nprompt sent: {final_prompt}\nLLM response:\n{llm_res}\n{'=' * 20}"
        )
        tool_used = False
        for tool, op in [
            ("add", operator.add),
            ("subtract", operator.sub),
            ("multiply", operator.mul),
        ]:
            pattern = f"<tool>{tool}\((-?\d+),\s?(-?\d+)\)</tool>"
            search = re.search(pattern, llm_res)
            if search:
                conversation.append(
                    f"<tool-result>{op(int(search.group(1)), int(search.group(2)))}</tool-result>"
                )
                tool_used = True
                break

        if tool_used:
            continue

        match = re.search(r"<answer>(-?\d+)</answer>", llm_res)
        if not match:
            final_prompt = "\n".join(conversation)
            logger.error("Could not find answer in response")
            logger.debug(f"{'=' * 20}\n{final_prompt}\n{'=' * 20}")
            return False
        llm_ans = int(match.group(1))
        return ans == llm_ans

    if num_iter == 3:
        final_prompt = "\n".join(conversation)
        logger.error("Agent got stuck in a loop")
        logger.debug(f"{'=' * 20}\n{final_prompt}\n{'=' * 20}")
        return False


def construct_equation() -> str:
    num_low = 20
    num_high = 250
    nums_low = 2
    nums_high = 3
    to_ret = ""
    nums = random.randint(nums_low, nums_high)
    mult_used = 0
    for i in range(nums):
        to_ret += str(random.randint(num_low, num_high))
        if i != nums - 1:
            to_ret += " * " if mult_used < 2 else random.choice([" + ", " - ", " * "])
            mult_used += 1
    return to_ret


def run(pre_prompt: str, equations: list[str]):
    correct = 0
    total = len(equations)
    for i in range(1, total + 1):
        if evaluate(pre_prompt, equations[i - 1]):
            correct += 1
        if i % 5 == 0:
            logger.info(f"Progress: {i}/{total} correct: ({correct}/{i})")
    print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    # logger.level = logging.DEBUG
    equations = [construct_equation() for _ in range(100)]
    run(SIMPLE_PROMPT, equations)
    run(AGENT_PROMPT, equations)
