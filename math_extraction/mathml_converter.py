import regex
import logging
import yaml
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from pathlib import Path
from collections import deque, OrderedDict
import lxml.etree as ET
from lxml.builder import E
from xml.sax.saxutils import escape
from .latex_parser import LaTeXParser, LaTeXToken, LaTeXTokenType

# Base classes (nếu không có module converters)
class ConversionContext:
    """Context for conversion process."""
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.metadata = {}

class BaseConverter:
    """Base converter class."""
    def __init__(self):
        self.context = ConversionContext()



logger = logging.getLogger(__name__)


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """LRU cache with time-based expiration."""
    def decorator(func):
        cache = OrderedDict()
        cache_time = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            if key in cache and current_time - cache_time[key] < seconds:
                cache.move_to_end(key)
                return cache[key]
                
            result = func(*args, **kwargs)
            
            cache[key] = result
            cache_time[key] = current_time
            
            if len(cache) > maxsize:
                oldest = next(iter(cache))
                del cache[oldest]
                del cache_time[oldest]
                
            return result
            
        wrapper.cache_clear = lambda: (cache.clear(), cache_time.clear())
        return wrapper
        
    return decorator


@dataclass
class MathMLElement:
    tag: str
    attributes: Dict[str, str] = field(default_factory=dict)
    content: Optional[str] = None
    children: List['MathMLElement'] = field(default_factory=list)
    
    def to_element(self) -> ET.Element:
        """Convert to lxml Element."""
        elem = ET.Element(self.tag, **self.attributes)
        
        if self.content is not None:
            elem.text = self.content
            
        for child in self.children:
            elem.append(child.to_element())
            
        return elem
    
    def to_xml(self, pretty_print: bool = True) -> str:
        """Convert to XML string."""
        elem = self.to_element()
        return ET.tostring(
            elem,
            pretty_print=pretty_print,
            encoding='unicode'
        )


class MathMLSymbolMapping:
    """Manage symbol mappings from configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.mappings = {}
        if config_path and config_path.exists():
            self._load_from_file(config_path)
        else:
            self._load_defaults()
            
    def _load_from_file(self, path: Path):
        """Load mappings from YAML/JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
                    
            self.mappings = data
            logger.info(f"Loaded symbol mappings from {path}")
            
        except Exception as e:
            logger.warning(f"Failed to load symbol mappings from {path}: {e}")
            self._load_defaults()
            
    def _load_defaults(self):
        """Load default symbol mappings."""
        self.mappings = {
            'operators': {
                '+': '+', '-': '−', '*': '⋅', '/': '/', '=': '=',
                '<': '&lt;', '>': '&gt;', '\\pm': '±', '\\mp': '∓',
                '\\times': '×', '\\div': '÷', '\\cdot': '⋅', '\\circ': '∘',
                '\\bullet': '•', '\\star': '⋆', '\\ast': '∗',
                '\\neq': '≠', '\\ne': '≠', '\\leq': '≤', '\\le': '≤',
                '\\geq': '≥', '\\ge': '≥', '\\ll': '≪', '\\gg': '≫',
                '\\subset': '⊂', '\\supset': '⊃', '\\subseteq': '⊆',
                '\\supseteq': '⊇', '\\in': '∈', '\\notin': '∉', '\\ni': '∋',
                '\\sim': '∼', '\\simeq': '≃', '\\approx': '≈', '\\cong': '≅',
                '\\equiv': '≡', '\\propto': '∝', '\\parallel': '∥', '\\perp': '⊥',
                '\\mid': '∣', '\\nmid': '∤', '\\vee': '∨', '\\wedge': '∧',
                '\\oplus': '⊕', '\\ominus': '⊖', '\\otimes': '⊗', '\\oslash': '⊘',
                '\\odot': '⊙', '\\cap': '∩', '\\cup': '∪', '\\sqcap': '⊓',
                '\\sqcup': '⊔', '\\vdash': '⊢', '\\dashv': '⊣', '\\models': '⊨',
                '\\therefore': '∴', '\\because': '∵', '\\emptyset': '∅',
                '\\infty': '∞', '\\partial': '∂', '\\nabla': '∇'
            },
            'greek_letters': {
                'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
                'epsilon': 'ε', 'varepsilon': 'ϵ', 'zeta': 'ζ', 'eta': 'η',
                'theta': 'θ', 'vartheta': 'ϑ', 'iota': 'ι', 'kappa': 'κ',
                'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ',
                'pi': 'π', 'varpi': 'ϖ', 'rho': 'ρ', 'varrho': 'ϱ',
                'sigma': 'σ', 'varsigma': 'ς', 'tau': 'τ', 'upsilon': 'υ',
                'phi': 'φ', 'varphi': 'ϕ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
                'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
                'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ',
                'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω'
            },
            'functions': {
                'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'log': 'log',
                'ln': 'ln', 'exp': 'exp', 'lim': 'lim', 'max': 'max', 'min': 'min'
            },
            'delimiters': {
                '(': '(', ')': ')', '[': '[', ']': ']',
                '\\{': '{', '\\}': '}', '\\langle': '⟨', '\\rangle': '⟩',
                '|': '|', '\\|': '‖'
            },
            'accents': {
                'hat': '^', 'tilde': '˜', 'bar': '¯', 'vec': '→',
                'dot': '˙', 'ddot': '¨', 'overline': '¯'
            }
        }
        
    def get_symbol(self, latex_symbol: str, category: Optional[str] = None) -> Optional[str]:
        """Get Unicode symbol for LaTeX command."""
        if category and category in self.mappings:
            return self.mappings[category].get(latex_symbol)
            
        for cat, mapping in self.mappings.items():
            if isinstance(mapping, dict) and latex_symbol in mapping:
                return mapping[latex_symbol]
                
        return None


class MathMLConverter:
    """Convert LaTeX to MathML with production-ready features."""
    
    MAX_INPUT_LENGTH = 10000
    MAX_CACHE_SIZE = 1000
    MAX_CACHE_ENTRY_SIZE = 5000
    CACHE_TTL = 3600  # 1 hour
    
    def __init__(self, symbol_config_path: Optional[Path] = None):
        self.parser = LaTeXParser()
        self.symbols = MathMLSymbolMapping(symbol_config_path)
        self._cache_stats = {'hits': 0, 'misses': 0}
        self._env_converters = {}
        self._register_default_converters()
        
    def _register_default_converters(self):
        """Register default environment converters."""
        pass
        
    @timed_lru_cache(seconds=CACHE_TTL, maxsize=MAX_CACHE_SIZE)
    def convert(self, latex: str, display_mode: bool = False, 
                validate: bool = True, optimize: bool = True) -> str:
        """Convert LaTeX to MathML with validation and optimization options."""
        if len(latex) > self.MAX_INPUT_LENGTH:
            raise ValueError(f"Input LaTeX too long: {len(latex)} > {self.MAX_INPUT_LENGTH}")
            
        try:
            structure = self.parser.parse(latex)
            
            if not structure.is_valid:
                logger.warning(f"LaTeX structure has errors: {structure.errors}")
                
            math_element = MathMLElement(
                tag='math',
                attributes={
                    'xmlns': 'http://www.w3.org/1998/Math/MathML',
                    'display': 'block' if display_mode else 'inline'
                }
            )
            
            if structure.tree['children']:
                context = ConversionContext(nodes=structure.tree['children'])
                content = self._convert_with_context(context)
                
                if content and content.children:
                    if len(content.children) == 1 and content.children[0].tag == 'mrow':
                        math_element.children = content.children[0].children
                    else:
                        math_element.children = content.children
                        
            result = math_element.to_xml()
            
            if validate:
                is_valid, errors = self.validate_mathml(result)
                if not is_valid:
                    logger.warning(f"Generated invalid MathML: {errors}")
                    
            if optimize:
                result = self.optimize_mathml(result)
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert LaTeX to MathML: {e}")
            return self._fallback_mathml(latex, display_mode)
            
    def _convert_with_context(self, context: ConversionContext) -> MathMLElement:
        """Convert nodes using context for efficient processing."""
        mrow = MathMLElement(tag='mrow')
        
        while context.has_more():
            node = context.peek()
            
            if node.get('type') == 'command':
                command_elem = self._handle_command(context)
                if command_elem:
                    mrow.children.append(command_elem)
                    
            elif node.get('type') in ['subscript', 'superscript']:
                context.consume()
                if mrow.children:
                    base = mrow.children.pop()
                else:
                    base = MathMLElement(tag='mi', content='')
                    
                script_elem = self._convert_script(node, base, context)
                mrow.children.append(script_elem)
                
            else:
                context.consume()
                mrow.children.append(self._convert_node(node))
                
        return mrow
        
    def _handle_command(self, context: ConversionContext) -> Optional[MathMLElement]:
        """Handle command with proper argument consumption."""
        node = context.consume()[0]
        value = node.get('value', '')
        metadata = node.get('metadata', {})
        
        if value == '\\frac':
            return self._handle_frac(context)
            
        elif value == '\\sqrt':
            return self._handle_sqrt(context, metadata)
            
        elif value == '\\text':
            return self._handle_text(context)
            
        elif metadata.get('subtype') == 'accent':
            return self._handle_accent(value, context)
            
        elif value == '\\binom':
            return self._handle_binom(context)
            
        else:
            return self._convert_command(node)
            
    def _handle_frac(self, context: ConversionContext) -> MathMLElement:
        """Handle \frac command."""
        frac = MathMLElement(tag='mfrac')
        
        numerator = self._get_next_argument(context)
        denominator = self._get_next_argument(context)
        
        frac.children.append(numerator)
        frac.children.append(denominator)
        
        return frac
        
    def _handle_sqrt(self, context: ConversionContext, metadata: Dict) -> MathMLElement:
        """Handle \\sqrt command with optional argument."""
        if metadata.get('optional_arg'):
            index = metadata['optional_arg']
            root = MathMLElement(tag='mroot')
            arg = self._get_next_argument(context)
            root.children.append(arg)
            root.children.append(MathMLElement(tag='mn', content=index))
            return root
        else:
            sqrt = MathMLElement(tag='msqrt')
            arg = self._get_next_argument(context)
            sqrt.children.append(arg)
            return sqrt
            
    def _handle_text(self, context: ConversionContext) -> MathMLElement:
        """Handle \text command."""
        arg = self._get_next_argument(context, as_text=True)
        return MathMLElement(tag='mtext', content=arg)
        
    def _handle_accent(self, command: str, context: ConversionContext) -> MathMLElement:
        """Handle accent commands."""
        accent_name = command.lstrip('\\')
        accent_symbol = self.symbols.get_symbol(accent_name, 'accents') or '^'
        
        base = self._get_next_argument(context)
        accent = MathMLElement(tag='mo', content=accent_symbol, 
                             attributes={'accent': 'true'})
        
        mover = MathMLElement(tag='mover')
        mover.children.append(base)
        mover.children.append(accent)
        
        return mover
        
    def _handle_binom(self, context: ConversionContext) -> MathMLElement:
        """Handle \binom command."""
        mrow = MathMLElement(tag='mrow')
        mrow.children.append(MathMLElement(tag='mo', content='('))
        
        frac = MathMLElement(tag='mfrac', attributes={'linethickness': '0'})
        frac.children.append(self._get_next_argument(context))
        frac.children.append(self._get_next_argument(context))
        mrow.children.append(frac)
        
        mrow.children.append(MathMLElement(tag='mo', content=')'))
        
        return mrow
        
    def _get_next_argument(self, context: ConversionContext, 
                          as_text: bool = False) -> MathMLElement:
        """Get next argument (single node or group)."""
        if not context.has_more():
            return MathMLElement(tag='mtext' if as_text else 'mi', content='')
            
        node = context.peek()
        
        if node.get('type') == 'group':
            context.consume()
            children = node.get('children', [])
            if as_text:
                text = ''.join(n.get('value', '') for n in children)
                return MathMLElement(tag='mtext', content=text)
            else:
                sub_context = ConversionContext(nodes=children)
                return self._convert_with_context(sub_context)
        else:
            context.consume()
            if as_text:
                return MathMLElement(tag='mtext', content=node.get('value', ''))
            else:
                return self._convert_node(node)
                
    def _convert_node(self, node: Dict[str, Any]) -> MathMLElement:
        """Convert single parser node to MathML element."""
        node_type = node.get('type')
        value = node.get('value', '')
        metadata = node.get('metadata', {})
        
        if node_type == 'variable':
            return MathMLElement(tag='mi', content=value)
            
        elif node_type == 'number':
            return MathMLElement(tag='mn', content=value)
            
        elif node_type == 'operator':
            if metadata.get('subtype') == 'function':
                func_name = value.lstrip('\\')
                func_text = self.symbols.get_symbol(func_name, 'functions') or func_name
                return MathMLElement(tag='mi', content=func_text)
                
            op_symbol = self.symbols.get_symbol(value, 'operators') or value
            return MathMLElement(tag='mo', content=op_symbol)
            
        elif node_type == 'symbol':
            if metadata.get('subtype') == 'greek':
                letter = value.lstrip('\\')
                symbol = self.symbols.get_symbol(letter, 'greek_letters') or letter
                return MathMLElement(tag='mi', content=symbol)
                
            return MathMLElement(tag='mi', content=value)
            
        elif node_type == 'delimiter':
            delim_symbol = self.symbols.get_symbol(value, 'delimiters') or value
            return MathMLElement(tag='mo', content=delim_symbol)
            
        elif node_type == 'command':
            return self._convert_command(node)
            
        elif node_type == 'group':
            children = node.get('children', [])
            context = ConversionContext(nodes=children)
            return self._convert_with_context(context)
            
        elif node_type == 'environment':
            return self._convert_environment(node)
            
        elif node_type == 'delimiter_pair':
            return self._convert_delimiter_pair(node)
            
        elif node_type == 'macro':
            return MathMLElement(tag='mi', content=value)
            
        elif node_type == 'spacing':
            return MathMLElement(tag='mspace', attributes={'width': '0.5em'})
            
        else:
            return MathMLElement(tag='mtext', content=str(value))
            
    def _convert_command(self, node: Dict[str, Any]) -> MathMLElement:
        """Convert LaTeX command to MathML."""
        value = node.get('value', '')
        metadata = node.get('metadata', {})
        
        if value in ['\\sum', '\\prod']:
            op_map = {'\\sum': '∑', '\\prod': '∏'}
            return MathMLElement(
                tag='mo',
                content=op_map.get(value, value),
                attributes={'largeop': 'true', 'movablelimits': 'true'}
            )
            
        elif value in ['\\int', '\\iint', '\\iiint', '\\oint']:
            int_map = {
                '\\int': '∫', '\\iint': '∬', 
                '\\iiint': '∭', '\\oint': '∮'
            }
            return MathMLElement(
                tag='mo',
                content=int_map.get(value, value),
                attributes={'largeop': 'true'}
            )
            
        elif metadata.get('subtype') == 'font':
            font_map = {
                'mathbf': 'bold', 'mathit': 'italic',
                'mathrm': 'normal', 'mathsf': 'sans-serif',
                'mathtt': 'monospace', 'mathcal': 'script',
                'mathbb': 'double-struck', 'mathfrak': 'fraktur'
            }
            
            font_cmd = value.lstrip('\\')
            variant = font_map.get(font_cmd, 'normal')
            
            return MathMLElement(
                tag='mstyle',
                attributes={'mathvariant': variant}
            )
            
        else:
            symbol = self.symbols.get_symbol(value)
            if symbol:
                return MathMLElement(tag='mo', content=symbol)
            else:
                return MathMLElement(tag='mi', content=value)
                
    def _convert_environment(self, node: Dict[str, Any]) -> MathMLElement:
        """Convert LaTeX environment to MathML."""
        env_name = node.get('name', '')
        children = node.get('children', [])
        
        env_info = self.symbols.mappings.get('environments', {}).get(env_name, {})
        
        if 'matrix' in env_name:
            return self._convert_matrix(env_name, children, env_info)
            
        elif env_name in ['align', 'aligned', 'alignat', 'alignedat', 'eqnarray']:
            return self._convert_align(children, env_info)
            
        elif env_name == 'cases':
            return self._convert_cases(children)
            
        elif env_name in ['gather', 'gathered']:
            return self._convert_gather(children)
            
        elif env_name == 'split':
            return self._convert_split(children)
            
        elif env_name in ['multline', 'multline*']:
            return self._convert_multline(children)
            
        elif env_name == 'array':
            return self._convert_array(node)
            
        else:
            context = ConversionContext(nodes=children)
            return self._convert_with_context(context)
            
    def _convert_matrix(self, matrix_type: str, children: List[Dict[str, Any]], 
                       env_info: Dict) -> MathMLElement:
        """Convert matrix environment to MathML."""
        mtable = MathMLElement(tag='mtable')
        
        delimiters = env_info.get('delimiters')
        if delimiters is None:
            delim_map = {
                'pmatrix': ('(', ')'),
                'bmatrix': ('[', ']'),
                'Bmatrix': ('{', '}'),
                'vmatrix': ('|', '|'),
                'Vmatrix': ('‖', '‖')
            }
            delimiters = delim_map.get(matrix_type.rstrip('*'))
            
        rows = self._split_by_delimiter(children, '\\\\')
        
        for row_nodes in rows:
            if not row_nodes:
                continue
                
            mtr = MathMLElement(tag='mtr')
            cells = self._split_by_delimiter(row_nodes, '&')
            
            for cell_nodes in cells:
                mtd = MathMLElement(tag='mtd')
                if cell_nodes:
                    context = ConversionContext(nodes=cell_nodes)
                    content = self._convert_with_context(context)
                    mtd.children.extend(content.children)
                mtr.children.append(mtd)
                
            mtable.children.append(mtr)
            
        if delimiters:
            mrow = MathMLElement(tag='mrow')
            mrow.children.append(MathMLElement(
                tag='mo',
                content=delimiters[0],
                attributes={'fence': 'true', 'stretchy': 'true'}
            ))
            mrow.children.append(mtable)
            mrow.children.append(MathMLElement(
                tag='mo',
                content=delimiters[1],
                attributes={'fence': 'true', 'stretchy': 'true'}
            ))
            return mrow
        else:
            return mtable
            
    def _convert_align(self, children: List[Dict[str, Any]], env_info: Dict) -> MathMLElement:
        """Convert align environment to MathML."""
        column_align = env_info.get('column_align', 'right left')
        
        mtable = MathMLElement(
            tag='mtable',
            attributes={'columnalign': column_align}
        )
        
        rows = self._split_by_delimiter(children, '\\\\')
        
        for row_nodes in rows:
            if not row_nodes:
                continue
                
            mtr = MathMLElement(tag='mtr')
            cells = self._split_by_delimiter(row_nodes, '&')
            
            for i, cell_nodes in enumerate(cells):
                align_parts = column_align.split()
                align = align_parts[i % len(align_parts)]
                
                mtd = MathMLElement(
                    tag='mtd',
                    attributes={'columnalign': align}
                )
                if cell_nodes:
                    context = ConversionContext(nodes=cell_nodes)
                    content = self._convert_with_context(context)
                    mtd.children.extend(content.children)
                mtr.children.append(mtd)
                
            mtable.children.append(mtr)
            
        return mtable
        
    def _convert_cases(self, children: List[Dict[str, Any]]) -> MathMLElement:
        """Convert cases environment to MathML."""
        mrow = MathMLElement(tag='mrow')
        mrow.children.append(MathMLElement(
            tag='mo',
            content='{',
            attributes={'fence': 'true', 'stretchy': 'true'}
        ))
        
        mtable = MathMLElement(
            tag='mtable',
            attributes={'columnalign': 'left left'}
        )
        
        rows = self._split_by_delimiter(children, '\\\\')
        
        for row_nodes in rows:
            if not row_nodes:
                continue
                
            mtr = MathMLElement(tag='mtr')
            cells = self._split_by_delimiter(row_nodes, '&')
            
            if len(cells) >= 2:
                mtd1 = MathMLElement(tag='mtd')
                context1 = ConversionContext(nodes=cells[0])
                content1 = self._convert_with_context(context1)
                mtd1.children.extend(content1.children)
                mtr.children.append(mtd1)
                
                mtd2 = MathMLElement(tag='mtd')
                mtext = MathMLElement(tag='mtext', content=' if ')
                mtd2.children.append(mtext)
                context2 = ConversionContext(nodes=cells[1])
                content2 = self._convert_with_context(context2)
                mtd2.children.extend(content2.children)
                mtr.children.append(mtd2)
            else:
                mtd = MathMLElement(tag='mtd')
                if cells and cells[0]:
                    context = ConversionContext(nodes=cells[0])
                    content = self._convert_with_context(context)
                    mtd.children.extend(content.children)
                mtr.children.append(mtd)
                
            mtable.children.append(mtr)
            
        mrow.children.append(mtable)
        
        return mrow
        
    def _convert_gather(self, children: List[Dict[str, Any]]) -> MathMLElement:
        """Convert gather environment to MathML."""
        mtable = MathMLElement(
            tag='mtable',
            attributes={'columnalign': 'center'}
        )
        
        rows = self._split_by_delimiter(children, '\\\\')
        
        for row_nodes in rows:
            if not row_nodes:
                continue
                
            mtr = MathMLElement(tag='mtr')
            mtd = MathMLElement(tag='mtd')
            
            if row_nodes:
                context = ConversionContext(nodes=row_nodes)
                content = self._convert_with_context(context)
                mtd.children.extend(content.children)
                
            mtr.children.append(mtd)
            mtable.children.append(mtr)
            
        return mtable
        
    def _convert_split(self, children: List[Dict[str, Any]]) -> MathMLElement:
        """Convert split environment to MathML."""
        return self._convert_align(children, {'column_align': 'right left'})
        
    def _convert_multline(self, children: List[Dict[str, Any]]) -> MathMLElement:
        """Convert multline environment to MathML."""
        mtable = MathMLElement(tag='mtable')
        
        rows = self._split_by_delimiter(children, '\\\\')
        
        for i, row_nodes in enumerate(rows):
            if not row_nodes:
                continue
                
            mtr = MathMLElement(tag='mtr')
            
            if i == 0:
                align = 'left'
            elif i == len(rows) - 1:
                align = 'right'
            else:
                align = 'center'
                
            mtd = MathMLElement(
                tag='mtd',
                attributes={'columnalign': align}
            )
            
            if row_nodes:
                context = ConversionContext(nodes=row_nodes)
                content = self._convert_with_context(context)
                mtd.children.extend(content.children)
                
            mtr.children.append(mtd)
            mtable.children.append(mtr)
            
        return mtable
        
    def _convert_array(self, node: Dict[str, Any]) -> MathMLElement:
        """Convert array environment to MathML."""
        children = node.get('children', [])
        metadata = node.get('metadata', {})
        
        column_spec = metadata.get('optional_arg', 'c')
        
        alignments = []
        for char in column_spec:
            if char == 'l':
                alignments.append('left')
            elif char == 'r':
                alignments.append('right')
            elif char == 'c':
                alignments.append('center')
                
        column_align = ' '.join(alignments) if alignments else 'center'
        
        mtable = MathMLElement(
            tag='mtable',
            attributes={'columnalign': column_align}
        )
        
        rows = self._split_by_delimiter(children, '\\\\')
        
        for row_nodes in rows:
            if not row_nodes:
                continue
                
            mtr = MathMLElement(tag='mtr')
            cells = self._split_by_delimiter(row_nodes, '&')
            
            for i, cell_nodes in enumerate(cells):
                align = alignments[i % len(alignments)] if alignments else 'center'
                
                mtd = MathMLElement(
                    tag='mtd',
                    attributes={'columnalign': align}
                )
                if cell_nodes:
                    context = ConversionContext(nodes=cell_nodes)
                    content = self._convert_with_context(context)
                    mtd.children.extend(content.children)
                mtr.children.append(mtd)
                
            mtable.children.append(mtr)
            
        return mtable
        
    def _convert_script(self, node: Dict[str, Any], base: MathMLElement,
                       context: ConversionContext) -> MathMLElement:
        """Convert subscript/superscript to MathML with lookahead."""
        script_type = node.get('type')
        children = node.get('children', [])
        
        script_context = ConversionContext(nodes=children)
        script_content = self._convert_with_context(script_context)
        
        next_node = context.peek()
        
        if (next_node and 
            next_node.get('type') in ['subscript', 'superscript'] and
            next_node.get('type') != script_type):
            
            context.consume()
            next_children = next_node.get('children', [])
            next_context = ConversionContext(nodes=next_children)
            next_content = self._convert_with_context(next_context)
            
            msubsup = MathMLElement(tag='msubsup')
            msubsup.children.append(base)
            
            if script_type == 'subscript':
                msubsup.children.append(script_content)
                msubsup.children.append(next_content)
            else:
                msubsup.children.append(next_content)
                msubsup.children.append(script_content)
                
            return msubsup
            
        else:
            tag = 'msub' if script_type == 'subscript' else 'msup'
            elem = MathMLElement(tag=tag)
            elem.children.append(base)
            elem.children.append(script_content)
            
            return elem
            
    def _convert_delimiter_pair(self, node: Dict[str, Any]) -> MathMLElement:
        """Convert \\left...\\right delimiter pair to MathML."""
        delimiter = node.get('delimiter', {})
        
        if delimiter.get('type') == 'null_delimiter' or delimiter.get('value') == '.':
            return MathMLElement(tag='mrow')
            
        delim_value = delimiter.get('value', '')
        delim_symbol = self.symbols.get_symbol(delim_value, 'delimiters') or delim_value
        
        return MathMLElement(
            tag='mo',
            content=delim_symbol,
            attributes={'stretchy': 'true', 'fence': 'true'}
        )
        
    def _split_by_delimiter(self, nodes: List[Dict[str, Any]], 
                           delimiter: str) -> List[List[Dict[str, Any]]]:
        """Split list of nodes by delimiter value (universal)."""
        result = [[]]
        
        for node in nodes:
            if node.get('value') == delimiter:
                result.append([])
            else:
                result[-1].append(node)
                
        return [group for group in result if group or len(result) == 1]
        
    def _fallback_mathml(self, latex: str, display_mode: bool) -> str:
        """Generate fallback MathML for failed conversions."""
        math_element = MathMLElement(
            tag='math',
            attributes={
                'xmlns': 'http://www.w3.org/1998/Math/MathML',
                'display': 'block' if display_mode else 'inline'
            }
        )
        
        merror = MathMLElement(tag='merror')
        mtext = MathMLElement(
            tag='mtext',
            content=f"Failed to convert: {latex[:100]}..."
        )
        
        merror.children.append(mtext)
        math_element.children.append(merror)
        
        return math_element.to_xml()
        
    @lru_cache(maxsize=256)
    def validate_mathml(self, mathml: str) -> Tuple[bool, List[str]]:
        """Validate MathML string using lxml."""
        errors = []
        
        try:
            parser = ET.XMLParser(recover=False)
            root = ET.fromstring(mathml.encode('utf-8'), parser)
            
            if not root.tag.endswith('math'):
                errors.append("Root element must be <math>")
                
            namespace = root.get('xmlns')
            if namespace != 'http://www.w3.org/1998/Math/MathML':
                errors.append("Missing or incorrect MathML namespace")
                
            self._validate_element(root, errors)
            
        except ET.XMLSyntaxError as e:
            errors.append(f"XML syntax error: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return len(errors) == 0, errors
        
    def _validate_element(self, element: ET.Element, errors: List[str]):
        """Recursively validate MathML element structure."""
        tag = ET.QName(element).localname
        
        valid_tags = {
            'math', 'mrow', 'mi', 'mn', 'mo', 'mtext', 'mspace',
            'ms', 'mglyph', 'msub', 'msup', 'msubsup', 'munder',
            'mover', 'munderover', 'mmultiscripts', 'mtable', 'mtr',
            'mtd', 'mlabeledtr', 'mstack', 'mlongdiv', 'msgroup',
            'msrow', 'mscarries', 'mscarry', 'msline', 'mfrac',
            'msqrt', 'mroot', 'mstyle', 'merror', 'mpadded',
            'mphantom', 'mfenced', 'menclose', 'maction',
            'semantics', 'annotation', 'annotation-xml', 'mprescripts',
            'none', 'maligngroup', 'malignmark'
        }
        
        if tag not in valid_tags:
            errors.append(f"Invalid MathML element: <{tag}>")
            
        content_elements = {'mi', 'mn', 'mo', 'mtext', 'ms'}
        if tag in content_elements and len(element) > 0:
            errors.append(f"Element <{tag}> should not have child elements")
            
        required_children = {
            'mfrac': 2, 'msub': 2, 'msup': 2, 'msubsup': 3,
            'munder': 2, 'mover': 2, 'munderover': 3, 'mroot': 2
        }
        
        if tag in required_children:
            expected = required_children[tag]
            actual = len(element)
            if actual != expected:
                errors.append(f"Element <{tag}> requires {expected} children, found {actual}")
                
        for child in element:
            self._validate_element(child, errors)
            
    def optimize_mathml(self, mathml: str) -> str:
        """Optimize MathML by removing redundant elements."""
        try:
            parser = ET.XMLParser(remove_blank_text=True)
            root = ET.fromstring(mathml.encode('utf-8'), parser)
            
            self._optimize_element(root)
            
            return ET.tostring(
                root,
                pretty_print=True,
                encoding='unicode'
            )
            
        except Exception as e:
            logger.warning(f"Failed to optimize MathML: {e}")
            return mathml
            
    def _optimize_element(self, element: ET.Element, parent: Optional[ET.Element] = None):
        """Recursively optimize MathML element."""
        for child in list(element):
            self._optimize_element(child, element)
            
        tag = ET.QName(element).localname
        
        if tag == 'mrow':
            if len(element) == 1 and parent is not None:
                child = element[0]
                index = list(parent).index(element)
                element.clear()
                parent.remove(element)
                parent.insert(index, child)
                
            elif len(element) == 0 and parent is not None:
                parent.remove(element)
                
        elif tag == 'mstyle':
            if not element.attrib and parent is not None:
                for child in list(element):
                    element.remove(child)
                    parent.insert(list(parent).index(element), child)
                parent.remove(element)
                
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': self._cache_stats['hits'] / max(1, self._cache_stats['hits'] + self._cache_stats['misses'])
        }
        
    def clear_cache(self):
        """Clear the conversion cache."""
        self.convert.cache_clear()
        self._cache_stats = {'hits': 0, 'misses': 0}


def latex_to_mathml(latex: str, display: bool = False,
                   validate: bool = True, optimize: bool = True,
                   config_path: Optional[Path] = None) -> str:
    """Convenience function to convert LaTeX to MathML."""
    converter = MathMLConverter(config_path)
    return converter.convert(latex, display, validate, optimize)


__all__ = ['MathMLConverter', 'MathMLElement', 'latex_to_mathml', 'ConversionContext']
