import warnings
from typing import Tuple, Set, List, Union

from lark import Transformer, v_args, Lark, Token, Tree

from npsem.stat_utils import ProbSpec
from npsem.utils import unique

__lark_parser_spec = """
    ?start: ( NEWLINE | exp )*
    ?exp: NAME "=" sum    -> assign_var
        | NAME "=" sum COMMENT    -> assign_var
    ?sum: product           
        | sum "+" product   -> add
        | sum "-" product   -> sub
    ?product: unary          
        | product "&" unary  -> and
        | product "|" unary  -> or_
        | product "^" unary  -> xor
        | product "⊕" unary  -> xor
        | product "<<" unary  -> lshift
        | product ">>" unary  -> rshift
    ?unary: atom            
         | "!" atom         -> not_
    ?atom: INT              -> intval
         | NAME             -> var
         | "(" sum ")"

    COMMENT: /#[^\\n]*/
    %import common.CNAME -> NAME
    %import common.INT
    %import common.WS_INLINE
    %import common.NEWLINE
    %ignore WS_INLINE
    %ignore COMMENT
"""


@v_args(inline=True)
class __CalculateTree(Transformer):
    # noinspection PyUnresolvedReferences
    from operator import add, sub, neg, or_, xor, not_, lshift, rshift
    # from operator import add, sub, neg, or_, xor, not_  # just in case if the above line is deleted...

    def __init__(self):
        super().__init__()
        self.vv = {}

    def assign_var(self, name, value):
        self.vv[name] = value
        return value

    def var(self, name):
        return self.vv[name]

    def update_values(self, vv):
        self.vv.update(vv)

    # noinspection PyMethodMayBeStatic
    def intval(self, num):
        return int(num)


def _tree_to_names(tree, names=None):
    if names is None:
        names = set()

    if isinstance(tree, Tree):
        for child in tree.children:
            if isinstance(child, Tree):
                _tree_to_names(child, names)
            elif isinstance(child, Token) and tree.data == 'var':
                names.add(str(child))

    return names


def parse_scm_functions(eqstr: str, return_edges=False):
    calc_parser = Lark(__lark_parser_spec, parser='lalr')
    var2line = dict()
    lines = eqstr.splitlines()

    edges = set()
    for line_no, s in enumerate(lines):
        tree = calc_parser.parse(s)
        if tree.data == 'assign_var':
            assigning_variable = str(tree.children[0])
            pas = _tree_to_names(tree.children[1])
            edges |= {(pa, assigning_variable) for pa in pas}
            var2line[assigning_variable] = line_no

    transformer = __CalculateTree()
    calc_functor = Lark(__lark_parser_spec, parser='lalr', transformer=transformer)

    def functor(V):
        def func(vv):
            transformer.update_values(vv)
            return calc_functor.parse(lines[var2line[V]])

        return func

    if return_edges:
        return {v: functor(v) for v in var2line}, edges
    else:
        return {v: functor(v) for v in var2line}


__ProbSpecLark = """
    ?start: "P" "(" outcomes ")"
        | "P" "(" outcomes "|" back ")"
    
    ?back:  dos
        | dos "," givens
        | givens
        
    ?dos: "do" "(" names ")" -> f_interventions
    
    ?outcomes: names -> f_outcomes
    
    ?givens: names  -> f_conditionals
    
    ?names: VAR ("," VAR)*

    VAR: /(?!(do)\\b)[a-z_][a-z0-9]*/i
    
    %import common.WS_INLINE
    %ignore WS_INLINE
"""


@v_args(inline=True)
class __ProbSpecTree(Transformer):

    def __init__(self):
        super().__init__()
        self.yy = set()
        self.xx = set()
        self.zz = set()

    def f_outcomes(self, name):
        if isinstance(name, Token):
            self.yy.add(str(name))
        else:
            self.yy |= {str(tok) for tok in name.children}

    def f_conditionals(self, name):
        if isinstance(name, Token):
            self.zz.add(str(name))
        else:
            self.zz |= {str(tok) for tok in name.children}

    def f_conditionals2(self, cond, intv):
        self.f_conditionals(cond)
        self.f_interventions(intv)

    def f_interventions(self, name):
        if isinstance(name, Token):
            self.xx.add(str(name))
        else:
            self.xx |= {str(tok) for tok in name.children}

    def to_ps(self, *_, **__) -> ProbSpec:
        return ProbSpec(self.xx, self.yy, self.zz)


def ensure_probspec(ps: Union[ProbSpec, str]) -> ProbSpec:
    if isinstance(ps, ProbSpec):
        return ps
    return parse_probspec(ps)


def parse_probspec(ps: str) -> ProbSpec:
    pst = __ProbSpecTree()
    Lark(__ProbSpecLark, parser='lalr', transformer=pst).parse(ps)
    return pst.to_ps()


__GraphLark = """
    ?start: chunk
        | start "," chunk
    ?chunk: atom
        | chunk "<->" atom    -> bidirected
        | chunk "->"  atom    -> to_right
        | chunk "<-"  atom    -> to_left
    ?atom: NAME               -> to_str
    
    DIGIT: "0".."9"
    LCASE_LETTER: "a".."z"
    UCASE_LETTER: "A".."Z"
    LETTER: UCASE_LETTER | LCASE_LETTER
    NAME: ("^"|"_"|LETTER|DIGIT)+

    %import common.WS_INLINE
    %ignore WS_INLINE
"""


@v_args(inline=True)
class __GraphMaker(Transformer):
    def __init__(self):
        super().__init__()
        self.Vs = set()
        self.edges = set()
        self.biedges = list()

    def to_str(self, name):
        self.Vs.add(str(name))
        return str(name)

    def bidirected(self, chunk, name):
        self.biedges.append((chunk, name))
        return name

    def to_right(self, chunk, name):
        self.edges.add((chunk, name))
        return name

    def to_left(self, chunk, name):
        self.edges.add((name, chunk))
        return name


def parse_graph(parsable: str, *, u_number_offset=0, u_names=None, u_name_prefix='U', u_name_postfix='', Vs=frozenset()) -> Tuple[Set[str], Set[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Parse chains of variables with edges between them into arguments for a causal diagram

    parsable: e.g., "A<->BB -> C_0 <- D^9 <- E <-> F -> G, H<-I<->J, K, L->M"
    """
    gm = __GraphMaker()
    graph_parser = Lark(__GraphLark, parser='lalr', transformer=gm).parse
    graph_parser(parsable)
    if u_names is not None:
        u_names = list(u_names)
        assert (aa := len(unibi := unique(gm.biedges))) <= (bb := len(u_names)), f"insufficient number of names provided {aa} <= {bb}"
        if aa < bb:
            warnings.warn(f'not all given names are used: {aa} < {bb}')
        bis = [(x, y, name) for (x, y), name in zip(unibi, u_names)]
    else:
        bis = [(x, y, f'{u_name_prefix}{i}{u_name_postfix}') for i, (x, y) in enumerate(unique(gm.biedges), start=u_number_offset)]
    return (gm.Vs | Vs), gm.edges, bis
