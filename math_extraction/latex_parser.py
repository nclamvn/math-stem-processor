import regex
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, cached_property
from collections import defaultdict
import hashlib
import threading


logger = logging.getLogger(__name__)


class LaTeXTokenType(Enum):
    COMMAND = "command"
    TEXT = "text"
    MATH = "math"
    ENVIRONMENT = "environment"
    DELIMITER = "delimiter"
    OPERATOR = "operator"
    SYMBOL = "symbol"
    NUMBER = "number"
    VARIABLE = "variable"
    SUBSCRIPT = "subscript"
    SUPERSCRIPT = "superscript"
    GROUP = "group"
    OPTIONAL = "optional"
    COMMENT = "comment"
    SPACING = "spacing"
    MACRO = "macro"


@dataclass
class LaTeXToken:
    type: LaTeXTokenType
    value: str
    position: int
    length: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.type.value, self.value, self.position))


@dataclass
class LaTeXStructure:
    tokens: List[LaTeXToken]
    tree: Dict[str, Any]
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    complexity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrieNode:
    __slots__ = ['children', 'is_end', 'value']
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None


class OperatorTrie:
    def __init__(self, operators: Set[str]):
        self.root = TrieNode()
        sorted_operators = sorted(operators, key=len, reverse=True)
        for op in sorted_operators:
            self._insert(op)
            
    def _insert(self, operator: str):
        node = self.root
        for char in operator:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.value = operator
        
    def find_longest_match(self, text: str, start: int) -> Optional[Tuple[str, int]]:
        node = self.root
        longest_match = None
        length = 0
        
        for i in range(start, min(len(text), start + 20)):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            length += 1
            if node.is_end:
                longest_match = (node.value, length)
                
        return longest_match


_operator_trie_instance = None
_operator_trie_lock = threading.Lock()


def get_operator_trie() -> OperatorTrie:
    global _operator_trie_instance
    if _operator_trie_instance is None:
        with _operator_trie_lock:
            if _operator_trie_instance is None:
                _operator_trie_instance = OperatorTrie(LaTeXParser.BINARY_OPERATORS)
    return _operator_trie_instance


class LaTeXParser:
    
    GREEK_LETTERS = {
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta',
        'eta', 'theta', 'vartheta', 'iota', 'kappa', 'lambda', 'mu',
        'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma',
        'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
        'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma',
        'Upsilon', 'Phi', 'Psi', 'Omega', 'ell', 'hbar', 'imath', 'jmath',
        'aleph', 'beth', 'gimel', 'daleth'
    }
    
    MATH_OPERATORS = {
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh', 'coth', 'log', 'ln', 'lg', 'exp',
        'lim', 'limsup', 'liminf', 'sup', 'inf', 'max', 'min',
        'det', 'dim', 'ker', 'rank', 'span', 'trace', 'deg',
        'gcd', 'lcm', 'Pr', 'arg', 'mod', 'bmod', 'pmod',
        'hom', 'ext', 'tor', 'im', 'Re', 'Im'
    }
    
    BINARY_OPERATORS = {
        '+', '-', '*', '/', '=', '<', '>', '\\pm', '\\mp', '\\times', '\\div',
        '\\cdot', '\\circ', '\\bullet', '\\star', '\\ast', '\\oplus', '\\ominus',
        '\\otimes', '\\oslash', '\\odot', '\\wedge', '\\vee', '\\cap', '\\cup',
        '\\neq', '\\ne', '\\leq', '\\le', '\\geq', '\\ge', '\\ll', '\\gg', 
        '\\subset', '\\supset', '\\subseteq', '\\supseteq', '\\in', '\\ni', 
        '\\notin', '\\sim', '\\simeq', '\\approx', '\\cong', '\\equiv', 
        '\\propto', '\\parallel', '\\perp', '\\mid', '\\nmid',
        '\\bigcup', '\\bigcap', '\\bigoplus', '\\bigotimes', '\\bigvee', '\\bigwedge',
        '\\coprod', '\\amalg', '\\sqcup', '\\sqcap', '\\dagger', '\\ddagger',
        '\\vdash', '\\dashv', '\\models', '\\smile', '\\frown'
    }
    
    DELIMITERS = {
        '(': ')', '[': ']', '\\{': '\\}', '\\langle': '\\rangle',
        '\\lfloor': '\\rfloor', '\\lceil': '\\rceil', '|': '|',
        '\\lvert': '\\rvert', '\\lVert': '\\rVert', '\\|': '\\|',
        '\\ulcorner': '\\urcorner', '\\llcorner': '\\lrcorner'
    }
    
    LEFT_RIGHT_DELIMITERS = {
        '(', ')', '[', ']', '\\{', '\\}', '|', '\\|', '.', 
        '\\langle', '\\rangle', '\\lfloor', '\\rfloor', '\\lceil', '\\rceil',
        '\\lvert', '\\rvert', '\\lVert', '\\rVert', '\\vert', '\\Vert',
        '\\uparrow', '\\downarrow', '\\updownarrow', '\\Uparrow', '\\Downarrow',
        '\\Updownarrow', '\\lgroup', '\\rgroup', '\\lmoustache', '\\rmoustache'
    }
    
    ACCENTS = {
        'hat', 'check', 'breve', 'acute', 'grave', 'tilde', 'bar',
        'vec', 'dot', 'ddot', 'dddot', 'overline', 'underline',
        'widehat', 'widetilde', 'overrightarrow', 'overleftarrow',
        'overbrace', 'underbrace', 'overarc', 'wideparen'
    }
    
    FONT_COMMANDS = {
        'mathrm', 'mathbf', 'mathit', 'mathsf', 'mathtt', 'mathcal',
        'mathscr', 'mathbb', 'mathfrak', 'boldsymbol', 'text', 'textbf',
        'textit', 'textrm', 'textsf', 'texttt', 'emph', 'textsc',
        'textsl', 'textup', 'textnormal'
    }
    
    SPACING_COMMANDS = {
        ',', ':', ';', '!', 'quad', 'qquad', 'hspace', 'vspace',
        'hfill', 'vfill', 'smallskip', 'medskip', 'bigskip',
        'thinspace', 'medspace', 'thickspace', 'negthinspace',
        'negmedspace', 'negthickspace', 'phantom', 'hphantom', 'vphantom'
    }
    
    def __init__(self):
        self.command_pattern = regex.compile(r'\\([a-zA-Z]+|[^a-zA-Z])')
        self.number_pattern = regex.compile(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
        self.variable_pattern = regex.compile(r'[a-zA-Z]')
        self.comment_pattern = regex.compile(r'(?:^|(?<=\s))(?<!\\)%.*$', regex.MULTILINE)
        
        sorted_operators = sorted(self.BINARY_OPERATORS, key=len, reverse=True)
        escaped_operators = [regex.escape(op) for op in sorted_operators]
        self.binary_operator_pattern = regex.compile('|'.join(escaped_operators))
        
        self._complexity_cache = {}
        self._known_commands = set()
        self._dynamic_macros = {}
        
    @cached_property
    def operator_trie(self) -> OperatorTrie:
        return get_operator_trie()
        
    def parse(self, latex: str, remove_comments: bool = True) -> LaTeXStructure:
        """Parse LaTeX string into structured representation.
        
        Args:
            latex: LaTeX string to parse
            remove_comments: Whether to remove comments before parsing
            
        Returns:
            LaTeXStructure with tokens, tree, validation results
        """
        original_latex = latex
        
        if remove_comments:
            latex, comments = self._extract_comments(latex)
        else:
            comments = []
            
        self._extract_macros(latex)
        
        tokens = self._tokenize(latex)
        
        if comments:
            for comment_pos, comment_text in comments:
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.COMMENT,
                    value=comment_text,
                    position=comment_pos,
                    length=len(comment_text),
                    metadata={'removed': remove_comments}
                ))
                
        tokens.sort(key=lambda t: t.position)
        
        tree = self._build_tree(tokens)
        errors, warnings = self._validate_structure(tokens, tree)
        complexity = self._calculate_complexity(tokens, tree)
        
        return LaTeXStructure(
            tokens=tokens,
            tree=tree,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            complexity_score=complexity,
            metadata={
                'original_length': len(original_latex),
                'token_count': len(tokens),
                'tree_depth': self._calculate_tree_depth(tree),
                'has_macros': bool(self._dynamic_macros),
                'comment_count': len(comments)
            }
        )
    
    def _extract_comments(self, latex: str) -> Tuple[str, List[Tuple[int, str]]]:
        """Extract and remove comments, returning clean text and comment positions."""
        comments = []
        
        for match in self.comment_pattern.finditer(latex):
            comments.append((match.start(), match.group()))
            
        clean_latex = self.comment_pattern.sub('', latex)
        
        return clean_latex, comments
    
    def _extract_macros(self, latex: str):
        """Extract user-defined macros from \\newcommand and \\def."""
        newcommand_pattern = regex.compile(
            r'\\newcommand\{\\([a-zA-Z]+)\}(?:\[(\d+)\])?\{([^}]+)\}'
        )
        def_pattern = regex.compile(r'\\def\\([a-zA-Z]+)\{([^}]+)\}')
        
        for match in newcommand_pattern.finditer(latex):
            macro_name = match.group(1)
            arg_count = int(match.group(2)) if match.group(2) else 0
            macro_body = match.group(3)
            
            self._dynamic_macros[macro_name] = {
                'args': arg_count,
                'body': macro_body
            }
            self._known_commands.add(macro_name)
            
        for match in def_pattern.finditer(latex):
            macro_name = match.group(1)
            macro_body = match.group(2)
            
            self._dynamic_macros[macro_name] = {
                'args': 0,
                'body': macro_body
            }
            self._known_commands.add(macro_name)
    
    def _tokenize(self, latex: str) -> List[LaTeXToken]:
        """Tokenize LaTeX string into list of tokens."""
        tokens = []
        pos = 0
        
        while pos < len(latex):
            if latex[pos] == '\\':
                token, new_pos = self._extract_command(latex, pos)
                tokens.append(token)
                pos = new_pos
                
            elif latex[pos] in '{}':
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.GROUP,
                    value=latex[pos],
                    position=pos,
                    length=1
                ))
                pos += 1
                
            elif latex[pos] == '[':
                if pos > 0 and tokens and tokens[-1].type == LaTeXTokenType.COMMAND:
                    token, new_pos = self._extract_optional_balanced(latex, pos)
                    tokens.append(token)
                    pos = new_pos
                else:
                    tokens.append(LaTeXToken(
                        type=LaTeXTokenType.DELIMITER,
                        value='[',
                        position=pos,
                        length=1
                    ))
                    pos += 1
                
            elif latex[pos] == '_':
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.SUBSCRIPT,
                    value='_',
                    position=pos,
                    length=1
                ))
                pos += 1
                
            elif latex[pos] == '^':
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.SUPERSCRIPT,
                    value='^',
                    position=pos,
                    length=1
                ))
                pos += 1
                
            elif latex[pos].isdigit() or (latex[pos] == '.' and pos + 1 < len(latex) and latex[pos + 1].isdigit()):
                match = self.number_pattern.match(latex, pos)
                if match:
                    tokens.append(LaTeXToken(
                        type=LaTeXTokenType.NUMBER,
                        value=match.group(),
                        position=pos,
                        length=len(match.group())
                    ))
                    pos = match.end()
                else:
                    pos += 1
                    
            elif latex[pos].isalpha():
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.VARIABLE,
                    value=latex[pos],
                    position=pos,
                    length=1
                ))
                pos += 1
                
            elif op_match := self.operator_trie.find_longest_match(latex, pos):
                op_value, op_length = op_match
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.OPERATOR,
                    value=op_value,
                    position=pos,
                    length=op_length
                ))
                pos += op_length
                
            elif latex[pos] in '()[]':
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.DELIMITER,
                    value=latex[pos],
                    position=pos,
                    length=1
                ))
                pos += 1
                
            elif latex[pos].isspace():
                space_start = pos
                while pos < len(latex) and latex[pos].isspace():
                    pos += 1
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.SPACING,
                    value=latex[space_start:pos],
                    position=space_start,
                    length=pos - space_start
                ))
                
            else:
                tokens.append(LaTeXToken(
                    type=LaTeXTokenType.SYMBOL,
                    value=latex[pos],
                    position=pos,
                    length=1
                ))
                pos += 1
                
        return tokens
    
    def _extract_command(self, latex: str, pos: int) -> Tuple[LaTeXToken, int]:
        """Extract command token and handle optional arguments."""
        match = self.command_pattern.match(latex, pos)
        if not match:
            return LaTeXToken(
                type=LaTeXTokenType.SYMBOL,
                value='\\',
                position=pos,
                length=1
            ), pos + 1
            
        command = match.group(1)
        full_command = match.group(0)
        end_pos = match.end()
        
        token_type = LaTeXTokenType.COMMAND
        metadata = {}
        
        if command in self.GREEK_LETTERS:
            token_type = LaTeXTokenType.SYMBOL
            metadata['subtype'] = 'greek'
        elif command in self.MATH_OPERATORS:
            token_type = LaTeXTokenType.OPERATOR
            metadata['subtype'] = 'function'
        elif command in self.FONT_COMMANDS:
            metadata['subtype'] = 'font'
        elif command in self.ACCENTS:
            metadata['subtype'] = 'accent'
        elif command in self.SPACING_COMMANDS:
            token_type = LaTeXTokenType.SPACING
            metadata['subtype'] = 'spacing'
        elif command in ['left', 'right']:
            metadata['subtype'] = 'delimiter_modifier'
        elif command in self._dynamic_macros:
            token_type = LaTeXTokenType.MACRO
            metadata['macro_info'] = self._dynamic_macros[command]
        elif command == 'begin' or command == 'end':
            token_type = LaTeXTokenType.ENVIRONMENT
            env_match = regex.match(r'\\(begin|end)\{(\w+\*?)\}', latex[pos:])
            if env_match:
                full_command = env_match.group(0)
                metadata['env_name'] = env_match.group(2)
                metadata['env_type'] = env_match.group(1)
                end_pos = pos + env_match.end()
                
        return LaTeXToken(
            type=token_type,
            value=full_command,
            position=pos,
            length=end_pos - pos,
            metadata=metadata
        ), end_pos
    
    def _extract_optional_balanced(self, latex: str, pos: int) -> Tuple[LaTeXToken, int]:
        """Extract balanced optional argument [...]."""
        if pos >= len(latex) or latex[pos] != '[':
            return LaTeXToken(
                type=LaTeXTokenType.SYMBOL,
                value='',
                position=pos,
                length=0
            ), pos
            
        depth = 1
        i = pos + 1
        content_start = i
        
        while i < len(latex) and depth > 0:
            if i > 0 and latex[i - 1] == '\\':
                i += 1
                continue
                
            if latex[i] == '[':
                depth += 1
            elif latex[i] == ']':
                depth -= 1
                
            i += 1
            
        if depth == 0:
            content = latex[content_start:i - 1]
            metadata = {'content': content}
            
            if '=' in content:
                parts = content.split(',')
                options = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        options[key.strip()] = value.strip()
                metadata['options'] = options
                
            return LaTeXToken(
                type=LaTeXTokenType.OPTIONAL,
                value=latex[pos:i],
                position=pos,
                length=i - pos,
                metadata=metadata
            ), i
        else:
            return LaTeXToken(
                type=LaTeXTokenType.DELIMITER,
                value='[',
                position=pos,
                length=1
            ), pos + 1
    
    def _build_tree(self, tokens: List[LaTeXToken]) -> Dict[str, Any]:
        """Build abstract syntax tree from tokens with support for nested scripts."""
        tree = {
            'type': 'root',
            'children': []
        }
        
        stack = [tree]
        script_stack = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            current = stack[-1]
            
            if token.type == LaTeXTokenType.SPACING and len(token.value.strip()) == 0:
                i += 1
                continue
                
            if token.type == LaTeXTokenType.GROUP and token.value == '{':
                group = {
                    'type': 'group',
                    'children': [],
                    'position': token.position
                }
                current['children'].append(group)
                stack.append(group)
                
            elif token.type == LaTeXTokenType.GROUP and token.value == '}':
                if len(stack) > 1:
                    stack.pop()
                else:
                    current['children'].append(self._token_to_node(token))
                    
            elif token.type == LaTeXTokenType.ENVIRONMENT:
                if token.metadata.get('env_type') == 'begin':
                    env = {
                        'type': 'environment',
                        'name': token.metadata.get('env_name'),
                        'children': [],
                        'position': token.position
                    }
                    current['children'].append(env)
                    stack.append(env)
                elif token.metadata.get('env_type') == 'end':
                    if len(stack) > 1 and stack[-1].get('type') == 'environment':
                        if stack[-1].get('name') == token.metadata.get('env_name'):
                            stack.pop()
                        else:
                            current['children'].append(self._token_to_node(token))
                    else:
                        current['children'].append(self._token_to_node(token))
                        
            elif token.type in [LaTeXTokenType.SUBSCRIPT, LaTeXTokenType.SUPERSCRIPT]:
                if not current['children']:
                    current['children'].append({
                        'type': 'empty',
                        'position': token.position
                    })
                    
                base_node = current['children'][-1]
                
                if base_node.get('type') in ['subscript', 'superscript']:
                    nested_script = {
                        'type': token.type.value,
                        'base': base_node,
                        'children': [],
                        'position': token.position,
                        'nested': True
                    }
                    current['children'][-1] = nested_script
                    script_node = nested_script
                else:
                    script_node = {
                        'type': token.type.value,
                        'base': base_node,
                        'children': [],
                        'position': token.position
                    }
                    current['children'][-1] = script_node
                
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if next_token.type == LaTeXTokenType.GROUP and next_token.value == '{':
                        group_depth = 1
                        j = i + 2
                        group_children = []
                        
                        while j < len(tokens) and group_depth > 0:
                            if tokens[j].type == LaTeXTokenType.GROUP:
                                if tokens[j].value == '{':
                                    group_depth += 1
                                elif tokens[j].value == '}':
                                    group_depth -= 1
                                    
                            if group_depth > 0:
                                group_children.append(self._token_to_node(tokens[j]))
                            j += 1
                            
                        script_node['children'] = group_children
                        i = j - 1
                    else:
                        script_node['children'].append(self._token_to_node(next_token))
                        i += 1
                        
            elif token.type == LaTeXTokenType.COMMAND and token.metadata.get('subtype') == 'delimiter_modifier':
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if next_token.value == '.':
                        delimiter_node = {
                            'type': 'delimiter_pair',
                            'modifier': token.value,
                            'delimiter': {'type': 'null_delimiter', 'value': '.'},
                            'position': token.position
                        }
                    else:
                        delimiter_node = {
                            'type': 'delimiter_pair',
                            'modifier': token.value,
                            'delimiter': self._token_to_node(next_token),
                            'position': token.position
                        }
                    current['children'].append(delimiter_node)
                    i += 1
                else:
                    current['children'].append(self._token_to_node(token))
                    
            else:
                current['children'].append(self._token_to_node(token))
                
            i += 1
            
        return tree
    
    def _token_to_node(self, token: LaTeXToken) -> Dict[str, Any]:
        """Convert token to tree node."""
        node = {
            'type': token.type.value,
            'value': token.value,
            'position': token.position
        }
        if token.metadata:
            node['metadata'] = token.metadata
        return node
    
    def _validate_structure(self, tokens: List[LaTeXToken], tree: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate LaTeX structure and return errors and warnings."""
        errors = []
        warnings = []
        
        brace_stack = []
        for i, token in enumerate(tokens):
            if token.type == LaTeXTokenType.GROUP:
                if token.value == '{':
                    brace_stack.append((i, token.position))
                elif token.value == '}':
                    if brace_stack:
                        brace_stack.pop()
                    else:
                        errors.append({
                            'type': 'unmatched_brace',
                            'message': 'Unmatched closing brace',
                            'position': token.position,
                            'token_index': i
                        })
                        
        for idx, pos in brace_stack:
            errors.append({
                'type': 'unmatched_brace',
                'message': 'Unmatched opening brace',
                'position': pos,
                'token_index': idx
            })
            
        delimiter_stack = []
        left_right_stack = []
        
        for i, token in enumerate(tokens):
            if token.type == LaTeXTokenType.COMMAND and token.metadata.get('subtype') == 'delimiter_modifier':
                if token.value == '\\left':
                    left_right_stack.append((token.value, token.position, i))
                elif token.value == '\\right':
                    if left_right_stack:
                        left_right_stack.pop()
                    else:
                        errors.append({
                            'type': 'unmatched_delimiter',
                            'message': 'Unmatched \\right',
                            'position': token.position,
                            'token_index': i
                        })
                        
            elif token.type == LaTeXTokenType.DELIMITER:
                if token.value in self.DELIMITERS:
                    delimiter_stack.append((token.value, token.position, i))
                elif delimiter_stack:
                    expected = self.DELIMITERS.get(delimiter_stack[-1][0])
                    if expected == token.value:
                        delimiter_stack.pop()
                        
        for delim, pos, idx in left_right_stack:
            errors.append({
                'type': 'unmatched_delimiter',
                'message': f'Unmatched {delim}',
                'position': pos,
                'token_index': idx
            })
            
        if delimiter_stack:
            for delim, pos, idx in delimiter_stack:
                warnings.append({
                    'type': 'unmatched_delimiter',
                    'message': f'Possibly unmatched delimiter: {delim}',
                    'position': pos,
                    'token_index': idx
                })
                
        env_stack = []
        for i, token in enumerate(tokens):
            if token.type == LaTeXTokenType.ENVIRONMENT:
                if token.metadata.get('env_type') == 'begin':
                    env_name = token.metadata.get('env_name')
                    env_stack.append((env_name, token.position, i))
                elif token.metadata.get('env_type') == 'end':
                    env_name = token.metadata.get('env_name')
                    if env_stack and env_stack[-1][0] == env_name:
                        env_stack.pop()
                    else:
                        errors.append({
                            'type': 'mismatched_environment',
                            'message': f'Mismatched environment: \\end{{{env_name}}}',
                            'position': token.position,
                            'token_index': i
                        })
                        
        for env_name, pos, idx in env_stack:
            errors.append({
                'type': 'unclosed_environment',
                'message': f'Unclosed environment: {env_name}',
                'position': pos,
                'token_index': idx
            })
            
        for i, token in enumerate(tokens):
            if token.type == LaTeXTokenType.OPTIONAL:
                options = token.metadata.get('options', {})
                for key, value in options.items():
                    if not value:
                        warnings.append({
                            'type': 'empty_option_value',
                            'message': f'Empty value for option: {key}',
                            'position': token.position,
                            'token_index': i
                        })
                        
        command_tokens = [t for t in tokens if t.type == LaTeXTokenType.COMMAND]
        unknown_commands = []
        
        known_commands = (
            self.GREEK_LETTERS | self.MATH_OPERATORS | self.FONT_COMMANDS |
            self.ACCENTS | self.SPACING_COMMANDS | self._known_commands |
            {'frac', 'sqrt', 'left', 'right', 'begin', 'end', 'limits', 'nolimits',
             'text', 'mbox', 'hbox', 'vbox', 'not', 'binom', 'choose', 'over',
             'atop', 'above', 'below', 'stackrel', 'overset', 'underset',
             'newcommand', 'def', 'let', 'renewcommand', 'providecommand'}
        )
        
        for i, token in enumerate(command_tokens):
            cmd = token.value.lstrip('\\')
            if cmd not in known_commands and not cmd.startswith('text'):
                unknown_commands.append((cmd, token.position, i))
                
        if unknown_commands:
            unique_cmds = {}
            for cmd, pos, idx in unknown_commands:
                if cmd not in unique_cmds:
                    unique_cmds[cmd] = (pos, idx)
                    
            for cmd, (pos, idx) in unique_cmds.items():
                warnings.append({
                    'type': 'unknown_command',
                    'message': f'Unknown command: \\{cmd}',
                    'position': pos,
                    'token_index': idx
                })
                
        return errors, warnings
    
    def _calculate_complexity(self, tokens: List[LaTeXToken], tree: Dict[str, Any]) -> float:
        """Calculate complexity score for LaTeX expression."""
        ordered_keys = [
            'tokens', 'depth', 'environments', 'fractions', 'integrals',
            'sums_products', 'matrices', 'scripts', 'roots', 'delimiters',
            'nested_scripts', 'macros', 'optionals'
        ]
        
        counts = {key: 0 for key in ordered_keys}
        
        counts['tokens'] = len(tokens)
        counts['depth'] = self._calculate_tree_depth(tree)
        
        for token in tokens:
            if token.type == LaTeXTokenType.ENVIRONMENT and token.metadata.get('env_type') == 'begin':
                counts['environments'] += 1
                if 'matrix' in token.metadata.get('env_name', ''):
                    counts['matrices'] += 1
            elif token.value == '\\frac':
                counts['fractions'] += 1
            elif token.value in ['\\int', '\\iint', '\\iiint', '\\oint']:
                counts['integrals'] += 1
            elif token.value in ['\\sum', '\\prod']:
                counts['sums_products'] += 1
            elif token.value == '\\sqrt':
                counts['roots'] += 1
            elif token.type in [LaTeXTokenType.SUBSCRIPT, LaTeXTokenType.SUPERSCRIPT]:
                counts['scripts'] += 1
            elif token.type == LaTeXTokenType.DELIMITER:
                counts['delimiters'] += 1
            elif token.type == LaTeXTokenType.MACRO:
                counts['macros'] += 1
            elif token.type == LaTeXTokenType.OPTIONAL:
                counts['optionals'] += 1
                
        def count_nested_scripts(node):
            nested = 0
            if isinstance(node, dict):
                if node.get('nested'):
                    nested += 1
                for child in node.get('children', []):
                    nested += count_nested_scripts(child)
            return nested
            
        counts['nested_scripts'] = count_nested_scripts(tree)
        
        cache_key = tuple(counts[k] for k in ordered_keys)
        if cache_key in self._complexity_cache:
            return self._complexity_cache[cache_key]
            
        complexity = 0.0
        
        weights = {
            'tokens': 0.05,
            'depth': 0.4,
            'environments': 0.8,
            'fractions': 0.6,
            'integrals': 1.0,
            'sums_products': 0.8,
            'matrices': 1.2,
            'scripts': 0.2,
            'roots': 0.4,
            'delimiters': 0.1,
            'nested_scripts': 0.5,
            'macros': 0.3,
            'optionals': 0.15
        }
        
        for key, weight in weights.items():
            complexity += counts[key] * weight
            
        complexity = min(10.0, complexity)
        self._complexity_cache[cache_key] = complexity
        
        return complexity
    
    def _calculate_tree_depth(self, node: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum depth of tree."""
        if 'children' not in node or not node['children']:
            return current_depth
            
        max_child_depth = current_depth
        for child in node['children']:
            if isinstance(child, dict):
                child_depth = self._calculate_tree_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
                
        return max_child_depth
    
    def extract_components(self, latex: str) -> Dict[str, List[str]]:
        """Extract various components from LaTeX expression."""
        structure = self.parse(latex)
        components = {
            'commands': [],
            'greek_letters': [],
            'operators': [],
            'variables': [],
            'numbers': [],
            'environments': [],
            'fractions': [],
            'matrices': [],
            'integrals': [],
            'sums_products': [],
            'delimiters': [],
            'accents': [],
            'macros': [],
            'optional_args': []
        }
        
        for token in structure.tokens:
            if token.type == LaTeXTokenType.COMMAND:
                components['commands'].append(token.value)
                if token.value == '\\frac':
                    components['fractions'].append(token.value)
                elif token.value in ['\\int', '\\iint', '\\iiint', '\\oint']:
                    components['integrals'].append(token.value)
                elif token.value in ['\\sum', '\\prod']:
                    components['sums_products'].append(token.value)
                elif token.metadata.get('subtype') == 'accent':
                    components['accents'].append(token.value)
                    
            elif token.type == LaTeXTokenType.SYMBOL and token.metadata.get('subtype') == 'greek':
                components['greek_letters'].append(token.value)
                
            elif token.type == LaTeXTokenType.OPERATOR:
                components['operators'].append(token.value)
                
            elif token.type == LaTeXTokenType.VARIABLE:
                components['variables'].append(token.value)
                
            elif token.type == LaTeXTokenType.NUMBER:
                components['numbers'].append(token.value)
                
            elif token.type == LaTeXTokenType.DELIMITER:
                components['delimiters'].append(token.value)
                
            elif token.type == LaTeXTokenType.MACRO:
                components['macros'].append(token.value)
                
            elif token.type == LaTeXTokenType.OPTIONAL:
                components['optional_args'].append(token.value)
                
            elif token.type == LaTeXTokenType.ENVIRONMENT and token.metadata.get('env_type') == 'begin':
                env_name = token.metadata.get('env_name')
                components['environments'].append(env_name)
                if 'matrix' in env_name:
                    components['matrices'].append(env_name)
                    
        for key in components:
            components[key] = list(dict.fromkeys(components[key]))
            
        return components
    
    def normalize_latex(self, latex: str, preserve_spacing: bool = False, aggressive: bool = False) -> str:
        """Normalize LaTeX expression for consistent formatting."""
        latex = regex.sub(r'\s+', ' ', latex.strip())
        
        latex = regex.sub(r'\\limits\s*', '', latex)
        latex = regex.sub(r'\\nolimits\s*', '', latex)
        
        if aggressive:
            latex = regex.sub(r'\\left\s*', '', latex)
            latex = regex.sub(r'\\right\s*', '', latex)
        
        if not preserve_spacing:
            if aggressive:
                latex = regex.sub(r'\\[,;:!]\s*', ' ', latex)
                latex = regex.sub(r'\\(quad|qquad)\s*', ' ', latex)
            else:
                latex = regex.sub(r'\\,(?![a-zA-Z])', ' ', latex)
                latex = regex.sub(r'\\,\s*(?=d[a-z])', r'\\,', latex)
                latex = regex.sub(r'(?<=[0-9])\s*\\,\s*(?=[0-9])', ' ', latex)
                
        latex = regex.sub(r'\s*([_^])\s*', r'\1', latex)
        
        latex = regex.sub(r'\s*([+\-*/=<>])\s*', r' \1 ', latex)
        
        latex = regex.sub(r'\s+', ' ', latex).strip()
        
        return latex
    
    def to_mathml(self, latex: str) -> Optional[str]:
        """Convert LaTeX to MathML using latex2mathml."""
        try:
            from latex2mathml import converter
            
            normalized = self.normalize_latex(latex, preserve_spacing=True)
            mathml = converter.convert(normalized)
            return mathml
        except ImportError:
            logger.error("latex2mathml not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to convert LaTeX to MathML: {e}")
            return None
    
    def to_sympy(self, latex: str) -> Optional[Any]:
        """Parse LaTeX expression with SymPy."""
        try:
            from sympy.parsing.latex import parse_latex
            from sympy import sympify
            
            normalized = self.normalize_latex(latex, aggressive=True)
            
            try:
                expr = parse_latex(normalized)
            except:
                expr = sympify(normalized)
                
            return expr
        except ImportError:
            logger.error("sympy not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse LaTeX with SymPy: {e}")
            return None
    
    def estimate_render_size(self, latex: str, base_font_size: float = 12.0, dpi: int = 96) -> Tuple[float, float]:
        """Estimate rendered size of LaTeX expression in pixels."""
        structure = self.parse(latex)
        components = self.extract_components(latex)
        
        char_width = base_font_size * 0.6 * (dpi / 96)
        char_height = base_font_size * 1.2 * (dpi / 96)
        
        effective_chars = sum(1 for t in structure.tokens if t.type not in [LaTeXTokenType.SPACING, LaTeXTokenType.COMMENT])
        base_width = effective_chars * char_width * 0.5
        base_height = char_height
        
        if components['fractions']:
            fraction_count = len(components['fractions'])
            base_height *= (2.0 + fraction_count * 0.2)
            base_width *= 0.9
            
        if components['matrices']:
            matrix_count = len(components['matrices'])
            base_height *= (2.5 + matrix_count * 0.5)
            base_width *= 1.5
            
        if components['integrals'] or components['sums_products']:
            base_height *= 1.8
            base_width *= 1.2
            
        for token in structure.tokens:
            if token.value in ['\\left', '\\right'] and token.metadata.get('subtype') == 'delimiter_modifier':
                base_height *= 1.1
                
        script_count = sum(1 for t in structure.tokens if t.type in [LaTeXTokenType.SUBSCRIPT, LaTeXTokenType.SUPERSCRIPT])
        if script_count > 0:
            base_height *= (1 + min(script_count * 0.2, 0.6))
            
        nested_scripts = structure.metadata.get('nested_scripts', 0)
        if nested_scripts > 0:
            base_height *= (1 + nested_scripts * 0.15)
            
        depth = structure.metadata.get('tree_depth', 1)
        base_height *= (1 + depth * 0.1)
        
        for cmd in components.get('accents', []):
            if cmd in ['\\widehat', '\\widetilde', '\\overrightarrow', '\\overleftarrow']:
                base_height *= 1.2
                
        return (round(base_width), round(base_height))
    
    def get_hash(self, latex: str) -> str:
        """Get hash of normalized LaTeX for deduplication."""
        normalized = self.normalize_latex(latex, aggressive=True)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def is_simple_expression(self, latex: str) -> bool:
        """Check if LaTeX expression is simple (no complex environments)."""
        structure = self.parse(latex)
        
        complex_indicators = [
            lambda s: s.metadata.get('tree_depth', 0) > 3,
            lambda s: any(t.type == LaTeXTokenType.ENVIRONMENT for t in s.tokens),
            lambda s: any(t.value in ['\\frac', '\\sqrt'] for t in s.tokens),
            lambda s: s.complexity_score > 2.0
        ]
        
        return not any(check(structure) for check in complex_indicators)


__all__ = [
    'LaTeXParser',
    'LaTeXToken',
    'LaTeXTokenType',
    'LaTeXStructure',
    'get_operator_trie'
]
