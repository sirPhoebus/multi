import unittest
import json
import re
from typing import Dict, Any

# Mocking the stripping logic from real_store.py
def mock_parse_llm_json(response: str) -> Dict[str, Any]:
    # 1. Strip <think>...</think> blocks if present
    clean_response = response
    if "<think>" in response:
        if "</think>" in response:
            clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        else:
            # Truncated inside think. Remove everything from <think> to end
            start_think = response.find("<think>")
            clean_response = response[:start_think].strip()

    # 2. Strip markdown code blocks
    clean_text = clean_response.replace("```json", "").replace("```", "").strip()
    
    # 3. Robust JSON extraction (find first { and last })
    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1:
        clean_text = clean_text[start:end+1]
    
    if not clean_text:
        return None

    try:
        return json.loads(clean_text)
    except:
        return None

class TestLLMParsing(unittest.TestCase):
    def test_truncated_think(self):
        # Scenario: Model was thinking in Chinese and got truncated before ever starting the JSON
        response = "<think>好的，我现在需要分析这篇论文并提取超参数。论文提到了PPO和一些学习率。我们来看一"
        result = mock_parse_llm_json(response)
        self.assertIsNone(result) # Correctly identifies that no JSON was found

    def test_complete_think_with_json(self):
        # Scenario: Model thought and then provided JSON
        response = "<think>Reasoning...</think> ```json {\"algorithm\": \"PPO\", \"hyperparameters\": {}, \"env_id\": \"CartPole-v1\"} ```"
        result = mock_parse_llm_json(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["algorithm"], "PPO")

    def test_truncated_json_repair_logic(self):
        # Scenario: Model provided JSON but got truncated at the end (handled by repair in real_store, but let's test extraction)
        response = "Here is the config: {\"algorithm\": \"SAC\", \"hyperparameters\": {\"lr\": 0.001}"
        # Note: the mock above doesn't have the repair logic (which adds '}}'), but it should find the bracket
        result = mock_parse_llm_json(response)
        # Without repair it fails json.loads, which is expected for this specific mock test
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
