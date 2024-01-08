import ast

def add(a, b):
    return a + b
print(add(3, 2))

# 获取函数的源代码
source = '''
def add(a, b):
    return a + b
'''
# 解析源代码为AST
parsed_source = ast.parse(source)
print(parsed_source)
print(ast.dump(parsed_source, indent=4))

class FunctionVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        print(f"Function name: {node.name}")
        self.generic_visit(node)

visitor = FunctionVisitor()
visitor.visit(parsed_source)

class RewriteAdd(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            return ast.BinOp(left=node.left, op=ast.Sub(), right=node.right)
        return self.generic_visit(node)

modified_tree = RewriteAdd().visit(parsed_source)
# 修复AST节点并添加缺失的行号信息
ast.fix_missing_locations(modified_tree)
exec(compile(modified_tree, filename="<ast>", mode="exec"))
print(add(3, 2))

