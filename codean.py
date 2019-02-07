#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import jedi
from sys import getsizeof

def autocomp(src, namespace, lineno, column:):
    return jedi.Interpreter(src, [globals()], line=len(src.splitlines()), column=len(src.splitlines()[-1])).completions()

# Compileable: ast.Module([node for node in ast.parse(source).body if node.lineno <= 6])

## Print globals (vars only)
#for k, v in globals().copy().items():
#    if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
#        print(k, type(v), repr(v)[:80], sep='\t')

#sys.getsizeof

MAX_VAR_SIZE=4096

_vars = defaultdict(lambda: defaultdict(list))
def dump_vars(v, lineno):
    global _vars
    for k, v in v.copy().items():
        if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
            _vars[lineno][k].append(v if getsizeof(v) <= MAX_VAR_SIZE else '<LARGE>')
    #print('dump_vars', lineno)

#dump ast.parse('dump_vars(locals(), n.lineno)').body[0]

# Дамп _до_ каждой строчки
class Tracer2(ast.NodeTransformer):
    def __init__(self, **kvargs):
        self._lines_done = []
        super(Tracer2, self).__init__(**kvargs)

    def visit(self, node):
        def tracer_ast(lineno):
            return ast.Expr(
                value=ast.Call(func=ast.Name(id='dump_vars', ctx=ast.Load()),
                args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]), ast.Num(n.lineno)],
                keywords=[]))

        if hasattr(node, 'body'):
            new_body = []
            prev_lineno = -1
            for i, n in enumerate(node.body):
                if n.lineno not in self._lines_done and n.lineno > prev_lineno:
                    self._lines_done.append(n.lineno)
                    prev_lineno = n.lineno
                    new_body.append(tracer_ast(n.lineno))
                new_body.append(n)
            node.body = new_body
        return self.generic_visit(node)
    
#tree = ast.parse(source)
#Tracer2().visit(tree)
#ast.fix_missing_locations(tree)
#c = compile(tree, '<ast>', 'exec')
#print(astor.to_source(tree))
