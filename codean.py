#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
from io import StringIO
from sys import getsizeof, exc_info #, modules
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

import jedi

_MAX_VAR_SIZE = 4096
jedi.settings.add_bracket_after_function = True
jedi.settings.case_insensitive_completion = False
jedi.preload_module('ourturtle')

from astor import dump_tree, to_source

def autocomp(src, namespace, lineno, column):
    return jedi.Interpreter(
        'from ourturtle import Turtle\n' + src, [namespace],
        line=lineno + 1 + 1,
        column=column).completions()

COMMON_CODE='<common>'

# Compileable: ast.Module([node for node in ast.parse(source).body if node.lineno <= 6])

## Print globals (vars only)
#for k, v in globals().copy().items():
#    if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
#        print(k, type(v), repr(v)[:80], sep='\t')

#sys.getsizeof

_vars = defaultdict(lambda: defaultdict(list))


def dump_vars(v, lineno):
    global _vars
    for k, v in v.copy().items():
        if any([isinstance(v, t)
                for t in [int, float, str, dict, tuple]]) and k[0] != '_':
            _vars[lineno][k].append(
                v if getsizeof(v) <= MAX_VAR_SIZE else '<LARGE>')
    # print('dump_vars', lineno)


def compare_ast(node1, node2):
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        if len(node1) == len(node2):
            return all(map(compare_ast, node1, node2))
        return False
    else:
        return node1 == node2


# dump ast.parse('dump_vars(locals(), n.lineno)').body[0]


# Дамп _до_ каждой строчки
class Tracer2(ast.NodeTransformer):
    def __init__(self, **kvargs):
        self._lines_done = []
        super(Tracer2, self).__init__(**kvargs)

    def visit(self, node):
        def tracer_ast(lineno):
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='dump_vars', ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id='locals', ctx=ast.Load()),
                            args=[],
                            keywords=[]),
                        ast.Num(n.lineno)
                    ],
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


class Break(Exception):
    pass


class AddBreak(ast.NodeTransformer):
    def __init__(self, lineno, **kvargs):
        self.lineno = lineno
        self.done = False
        super(AddBreak, self).__init__(**kvargs)

    def visit(self, node):
        # print('lineno', getattr(node, 'lineno', None))
        # try:
        #     print(repr(astor.to_source(node)))
        # except:
        #     print('---')
        def break_ast(lineno):
            return ast.Raise(
                exc=ast.Name(id='Break', ctx=ast.Load()), cause=None)

        if hasattr(node, 'body'):
            new_body = []
            for i, child in enumerate(node.body):
                if not self.done and child.lineno >= self.lineno:
                    self.done = True
                    newchild = ast.copy_location(
                        break_ast(child.lineno), child)
                    new_body.append(newchild)
                new_body.append(child)
            node.body = new_body
        return self.generic_visit(node) if not self.done else node


class VarLister(ast.NodeVisitor):
    def __init__(self):
        self.vars_store = set()
        self.vars_load = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.vars_load.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.vars_store.add(node.id)
        self.generic_visit(node)


class CodeRunner:

    def __init__(self, source=None, name='<code-input>', globals={},
                 special_funcs=[]):
        self._name = name
        self._special_funcs = special_funcs
        self._ast = {}
        self._codeobjs = {}
        self._common_vars = set()
        self._globals = {}
        self.default_globals = {'Break': Break}
        self.default_globals.update(globals)
        self.text_stream = StringIO()
        self.reset(source)

    def reset(self, source=None, globals={}):
        self._breakpoint = None
        self.set_globals(globals, False)
        self.text_stream.close()
        self.text_stream = StringIO()
        if source:
            self.parse(source)

    def set_globals(self, globals, update=True):
        if not update:
            self._globals.clear()
            self._globals.update(self.default_globals)
        self._globals.update(globals)

    @property
    def globals(self):
        return self._globals

    @property
    def common_vars(self):
        return self._common_vars

    # @globals.setter
    # def globals(self, globals):
    #     self.set_globals(globals, False)

    def exception_lineno(self, exc_obj):
        exc_tb = exc_obj.__traceback__
        lineno = None
        if hasattr(exc_obj, 'lineno'):
            # print('exc_obj', exc_obj.lineno, exc_obj.filename)
            lineno = exc_obj.lineno

        while exc_tb:
            if exc_tb.tb_frame.f_code.co_filename == self._name:
                lineno = exc_tb.tb_lineno
            exc_tb = exc_tb.tb_next

        return lineno

    def set_breakpoint(self, breakpoint):
        changed = self.parse(self._source, breakpoint)
        self.compile(changed)
        return changed

    def parse(self, source, breakpoint=None):
        tree = ast.parse(source)
        # print(source)
        # print(' %' * 20)
        # print(to_source(tree))
        self._source = source
        self._breakpoint = breakpoint
        changed = []

        if breakpoint:
            AddBreak(breakpoint).visit(tree)
            ast.fix_missing_locations(tree)

        new_body = []
        new_ast = {}
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name in self._special_funcs:
                new_ast[node.name] = ast.Module(body=[node])
                if not compare_ast(self._ast.get(node.name, None), new_ast[node.name]):
                    changed.append(node.name)
            else:
                new_body.append(node)
        tree.body = new_body
        if not compare_ast(self._ast.get(COMMON_CODE, None), tree):
            changed = [COMMON_CODE] + changed  # Preserving order: common first.
        new_ast[COMMON_CODE] = tree
        self._ast = new_ast

        vl = VarLister()
        vl.visit(tree)
        self._common_vars = vl.vars_load

        return changed

    def compile(self, parts=None, **kvargs):
        codeobjs = self._codeobjs
        for p in parts or self._ast.keys():
            codeobjs[p] = compile(self._ast[p], self._name, 'exec', **kvargs)

    def execute(self, parts=None):
        if self.text_stream.closed:
            self.text_stream = StringIO()
        cobjs = self._codeobjs
        if not parts:  # Preserving order: common first.
            parts = [COMMON_CODE] + list(set(cobjs.keys()) - set([COMMON_CODE]))
        for p in parts:
            try:
                with redirect_stdout(self.text_stream):
                    with redirect_stderr(self.text_stream):
                        exec(cobjs[p], self._globals)

            except Break:
                print("* Break *")

    def call(self, func, *args, **kvargs):
        # if func not in self._globals:
            # raise NameError(f"name {repr(func)} is not defined")
        try:
            with redirect_stdout(self.text_stream):
                with redirect_stderr(self.text_stream):
                    return self._globals[func](*args, **kvargs)
        except Break:
            print("* Break *")

    def call_if_exists(self, func, *args, **kvargs):
        if func in self._globals:
            return self.call(func, *args, **kvargs)

        # try:
        #     break_at = None
        #     if self.run_to_cursor:
        #         break_at = self.init_editor.cursor_row + 2
        #         print('cursor_row', self.init_editor.cursor_row)

        #     self._code_compiled = compile(
        #         tree,
        #         self._name,
        #         break_at=break_at)

        # except Exception as e:
        #     print('E:', e)
        #     print('* ' * 40)
        #     exc_type, exc_obj, exc_tb = exc_info()
        #     line_num = None
        #     if hasattr(exc_obj, 'lineno'):
        #         print('exc_obj', exc_obj.lineno, exc_obj.filename)
        #         line_num = exc_obj.lineno

        #     while exc_tb:
        #         if exc_tb.tb_frame.f_code.co_filename == self._name:
        #             line_num = exc_tb.tb_lineno
        #         exc_tb = exc_tb.tb_next

        #     self.init_editor.highlight_line(line_num)
        # else:
        #     self.init_editor.highlight_line(None)
        #     self.trigger_exec_init()



# https://greentreesnakes.readthedocs.io/en/latest/
# https://www.mattlayman.com/blog/2018/decipher-python-ast/
# https://github.com/berkerpeksag/astor
