#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import astunparse
from typing import List, Dict, Any, Tuple, Optional

from .executor_utils import function_with_timeout
from .executor_types import ExecuteResult, Executor


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    
def execute_code_get_return(code: str):
    local_vars = {}
    try:
        exec(code, {}, local_vars)
        if 'answer' in local_vars:
            return local_vars['answer']
        else:
            return None
    except Exception as e:
        return f"Error occurred: {e}"

class PyExecutor(Executor):
    def __init__(self, get_answer_expr: str = "solution()"):
        self.get_answer_expr = get_answer_expr

    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests.append(tests[i])
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests.append(f"{tests[i]} # output: {output}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False

def run_python_code(code: str, timeout: int = 10) -> str:
    """
    Executes Python code in a subprocess and captures standard output and standard error.
    Used natively by LLM tool calling.
    """
    import tempfile
    import subprocess
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
        
    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}"
            
        if not output.strip():
            output = "[Code executed successfully with no output]"
            
        return output
    except subprocess.TimeoutExpired:
        return f"[Error: Code execution timed out after {timeout} seconds]"
    except Exception as e:
        return f"[Execution Error: {str(e)}]"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        