import os
import ast
import networkx as nx

def get_module_name(filepath: str, root_dir: str) -> str:
    """Convert file path to module name (without .py)."""
    rel = os.path.relpath(filepath, root_dir)
    return rel.replace(os.sep, ".").replace(".py", "")

def parse_imports(tree: ast.AST, module: str):
    """Extract import alias mappings from a file."""
    alias_map = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module:  # e.g., from base import BaseModel
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    alias_map[imported_name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.asname or alias.name
                alias_map[imported_name] = alias.name  # may need resolving later
    return alias_map

def build_code_graph(code_dir: str):
    graph = nx.DiGraph()

    for root, _, files in os.walk(code_dir):
        for file in files:
            if file.endswith(".py") and ".ipynb_checkpoints" not in root:
                path = os.path.join(root, file)
                module = get_module_name(path, code_dir)

                with open(path, "r", encoding="utf-8") as f:
                    try:
                        source = f.read()
                        tree = ast.parse(source)
                    except SyntaxError:
                        continue

                alias_map = parse_imports(tree, module)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = f"{module}.{node.name}"
                        docstring = ast.get_docstring(node) or ""

                        graph.add_node(
                            class_name,
                            type="class",
                            file=path,
                            docstring=docstring,
                        )

                        # Handle inheritance
                        for base in node.bases:
                            if isinstance(base, ast.Name):  # BaseModel
                                base_name = alias_map.get(base.id, base.id)
                                graph.add_edge(base_name, class_name, relation="inherits")
                            elif isinstance(base, ast.Attribute):  # e.g., models.BaseModel
                                base_name = f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr
                                graph.add_edge(base_name, class_name, relation="inherits")

                        # Handle methods inside class
                        for func in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                            func_name = f"{class_name}.{func.name}"
                            fdoc = ast.get_docstring(func) or ""
                            graph.add_node(
                                func_name,
                                type="method",
                                file=path,
                                docstring=fdoc,
                            )
                            graph.add_edge(class_name, func_name, relation="has_method")

                    elif isinstance(node, ast.FunctionDef):
                        func_name = f"{module}.{node.name}"
                        docstring = ast.get_docstring(node) or ""
                        graph.add_node(
                            func_name,
                            type="function",
                            file=path,
                            docstring=docstring,
                        )

    return graph
