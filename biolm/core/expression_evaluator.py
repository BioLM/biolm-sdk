"""Template expression evaluation for protocol outputs.

This module provides safe evaluation of template expressions like `${{ field_name }}`
and `${{ score > 0.5 }}` used in protocol output configurations.
"""
import operator
from typing import Any, Dict, Tuple, Union


def extract_template_expr(value: Any) -> Tuple[bool, str]:
    """Extract template expression from a value if present.
    
    Args:
        value: Value that may contain a template expression
        
    Returns:
        Tuple of (is_template_expr, expression_string)
        - is_template_expr: True if value is a template expression
        - expression_string: The inner expression (without ${{ }} wrapper)
    """
    if isinstance(value, str) and value.startswith("${{") and value.endswith("}}"):
        # Extract inner expression
        inner = value[3:-2].strip()
        return True, inner
    return False, ""


def evaluate_expression(expr: str, context: Dict[str, Any]) -> Any:
    """Evaluate a template expression safely.
    
    Args:
        expr: The expression to evaluate (without ${{ }} wrapper)
        context: Dictionary of variables available in the expression
        
    Returns:
        The evaluated result
        
    Raises:
        ValueError: If expression evaluation fails
        KeyError: If a required variable is missing from context
    """
    # Create safe evaluation context
    safe_globals = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "round": round,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "None": None,
            "True": True,
            "False": False,
        },
        # Math operations
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }
    
    # Add operators
    safe_globals.update({
        "operator": operator,
    })
    
    # Context becomes locals
    safe_locals = context.copy()
    
    try:
        # Evaluate the expression
        result = eval(expr, safe_globals, safe_locals)
        return result
    except NameError as e:
        # Provide helpful error message
        missing_var = str(e).split("'")[1] if "'" in str(e) else "unknown"
        raise KeyError(
            f"Variable '{missing_var}' not found in context. "
            f"Available variables: {list(context.keys())}"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expr}': {e}") from e


def evaluate_where_clause(expr: str, row: Dict[str, Any]) -> bool:
    """Evaluate a where clause expression row-by-row.
    
    Args:
        expr: The filter expression (may include ${{ }} wrapper)
        row: The row data to evaluate against
        
    Returns:
        True if row matches the filter, False otherwise
        
    Raises:
        ValueError: If expression evaluation fails
    """
    # Extract template expression if present
    is_template, inner_expr = extract_template_expr(expr)
    if is_template:
        expr = inner_expr
    
    try:
        result = evaluate_expression(expr, row)
        # Convert to boolean
        if isinstance(result, bool):
            return result
        # Truthy/falsy evaluation
        return bool(result)
    except (KeyError, ValueError) as e:
        # If a field is missing, treat as False (row doesn't match)
        # But log the error for debugging
        return False


def evaluate_template_value(value: Any, context: Dict[str, Any]) -> Any:
    """Evaluate a value that may contain a template expression.
    
    Args:
        value: Value that may be a template expression or literal
        context: Dictionary of variables available for evaluation
        
    Returns:
        The evaluated value (or original value if not a template expression)
    """
    is_template, expr = extract_template_expr(value)
    if is_template:
        return evaluate_expression(expr, context)
    return value


