import ast
import subprocess
import os
import shutil
import time
import tempfile
from typing import Tuple, Optional

class CodeSandbox:
    """
    Safely validates and tests Python code before allowing it to be applied 
    to the agent's codebase.
    """
    def __init__(self, sandbox_dir: str = "sandbox_env"):
        self.sandbox_dir = sandbox_dir
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Checks if the code is valid Python syntax."""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"

    def run_unit_test(self, code: str, test_script_content: str, timeout: int = 5) -> Tuple[bool, str]:
        """
        Runs the proposed code against a temporary test script in a subprocess.
        Returns: (passed, output_log)
        """
        # 1. Create a temp directory for this run
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = os.path.join(temp_dir, "candidate_module.py")
            test_path = os.path.join(temp_dir, "test_candidate.py")
            
            # 2. Write the candidate code
            with open(module_path, "w") as f:
                f.write(code)
                
            # 3. Write the test script
            # The test script must import 'candidate_module'
            with open(test_path, "w") as f:
                f.write(test_script_content)
                
            # 4. Execute the test
            try:
                # We run python test_candidate.py inside temp_dir
                # PYTHONPATH needs to include temp_dir
                env = os.environ.copy()
                env["PYTHONPATH"] = temp_dir + os.pathsep + env.get("PYTHONPATH", "")
                
                result = subprocess.run(
                    ["python", "test_candidate.py"],
                    cwd=temp_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    return True, f"Tests Passed.\nOutput: {result.stdout}"
                else:
                    return False, f"Tests Failed.\nStderr: {result.stderr}\nStdout: {result.stdout}"
                    
            except subprocess.TimeoutExpired:
                return False, f"Test timed out after {timeout} seconds."
            except Exception as e:
                return False, f"Execution failed: {e}"
    
    def apply_patch(self, target_file: str, new_content: str, backup: bool = True) -> bool:
        """
        Overwrites target_file with new_content. 
        Creates a .bak file if backup=True.
        """
        try:
            if backup and os.path.exists(target_file):
                shutil.copy(target_file, target_file + ".bak")
                
            with open(target_file, "w") as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"[Sandbox] Failed to apply patch: {e}")
            return False
