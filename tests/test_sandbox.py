from marl_scientist.sandbox.sandbox import CodeSandbox
import os

def test_sandbox():
    print("Testing CodeSandbox...")
    sandbox = CodeSandbox()
    
    # 1. Test Valid Syntax
    good_code = "def hello(): return 'world'"
    valid, msg = sandbox.validate_syntax(good_code)
    assert valid, f"Valid code failed: {msg}"
    print("[PASS] Syntax Check (Valid)")
    
    # 2. Test Invalid Syntax
    bad_code = "def hello() return 'world'" # Missing colon
    valid, msg = sandbox.validate_syntax(bad_code)
    assert not valid, "Invalid code passed syntax check"
    print("[PASS] Syntax Check (Invalid)")
    
    # 3. Test Unit Test Execution (Success)
    candidate = """
def add(a, b):
    return a + b
"""
    test_script = """
import unittest
from candidate_module import add

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

if __name__ == '__main__':
    unittest.main()
"""
    passed, out = sandbox.run_unit_test(candidate, test_script)
    assert passed, f"Unit test should pass. output: {out}"
    print("[PASS] Unit Test Execution (Success)")
    
    # 4. Test Unit Test Execution (Failure)
    candidate_bad = """
def add(a, b):
    return a - b # Bug
"""
    passed, out = sandbox.run_unit_test(candidate_bad, test_script)
    assert not passed, "Unit test should fail."
    print("[PASS] Unit Test Execution (Failure Rejection)")
    
    print("\nAll Sandbox tests passed!")

if __name__ == "__main__":
    test_sandbox()
