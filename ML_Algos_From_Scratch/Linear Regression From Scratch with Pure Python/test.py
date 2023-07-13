import ast
import astor
import re

def remove_docs_and_comments(file):
    with open(file,"r") as f:
        code = f.read()
    parsed = ast.parse(code)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            # set value to empty string
            node.value = ast.Constant(value='')
    formatted_code = astor.to_source(parsed)
    pattern = r'^.*"""""".*$' # remove empty """"""
    formatted_code = re.sub(pattern, '', formatted_code, flags=re.MULTILINE)
    return formatted_code

remove_docs_and_comments("Linear_Regression_From_Scratch.py")