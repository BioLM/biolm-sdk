"""Tests for protocol validation functionality."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from biolm.cli import cli
from biolm.protocols import (
    Protocol,
    ProtocolValidationResult,
    ValidationError,
)


@pytest.fixture
def valid_protocol_yaml(tmp_path):
    """Create a valid protocol YAML file for testing (server format: InputSpec, top-level progress/ranking)."""
    protocol_content = """
name: Test Protocol
schema_version: 1
description: A test protocol

inputs:
  sequence:
    type: text
    label: Protein sequence
    required: true
    initial: ''
    min_length: 1
    max_length: 2000
  n_samples:
    type: integer
    label: N samples
    optional: true
    initial: '10'
    min: 1
    max: 100
  temperature:
    type: float
    label: Temperature
    optional: true
    initial: '1.0'

progress:
  total_expected: 1

ranking:
  field: pdb
  order: descending
  top_n: 10

tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items:
        - sequence: ${{ sequence }}
      params:
        temperature: ${{ temperature }}
    response_mapping:
      pdb: "${{ response.results[*].pdb }}"
  - id: task2
    type: gather
    from: task1
    fields: [pdb]
    depends_on: [task1]
    into: 5
"""
    protocol_file = tmp_path / "valid_protocol.yaml"
    protocol_file.write_text(protocol_content)
    return str(protocol_file)


@pytest.fixture
def invalid_yaml_syntax(tmp_path):
    """Create a protocol file with invalid YAML syntax."""
    protocol_file = tmp_path / "invalid_syntax.yaml"
    protocol_file.write_text("name: Test\n  invalid: indentation: here")
    return str(protocol_file)


@pytest.fixture
def missing_schema_version(tmp_path):
    """Create a protocol file missing required 'inputs' (schema requires name, inputs, tasks)."""
    protocol_content = """
name: Test Protocol
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items: []
"""
    protocol_file = tmp_path / "missing_inputs.yaml"
    protocol_file.write_text(protocol_content)
    return str(protocol_file)


@pytest.fixture
def invalid_task_reference(tmp_path):
    """Create a protocol with invalid task ID references in depends_on."""
    protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  x:
    type: text
    label: X
    required: true
    initial: ''
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items: []
    depends_on: [nonexistent_task]
"""
    protocol_file = tmp_path / "invalid_ref.yaml"
    protocol_file.write_text(protocol_content)
    return str(protocol_file)


@pytest.fixture
def circular_dependency(tmp_path):
    """Create a protocol with circular task dependencies."""
    protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  x:
    type: text
    label: X
    required: true
    initial: ''
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items: []
    depends_on: [task2]
  - id: task2
    slug: esm2-8m
    action: encode
    request_body:
      items: []
    depends_on: [task1]
"""
    protocol_file = tmp_path / "circular.yaml"
    protocol_file.write_text(protocol_content)
    return str(protocol_file)


@pytest.fixture
def invalid_template_expression(tmp_path):
    """Create a protocol with invalid template expression (unclosed ${{)."""
    protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  x:
    type: text
    label: X
    required: true
    initial: ''
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items:
        - sequence: ${{ invalid_expression
"""
    protocol_file = tmp_path / "invalid_template.yaml"
    protocol_file.write_text(protocol_content)
    return str(protocol_file)


class TestValidationError:
    """Test ValidationError dataclass."""
    
    def test_validation_error_creation(self):
        """Test creating a ValidationError."""
        error = ValidationError(
            message="Test error",
            path="tasks[0].id",
            error_type="semantic"
        )
        assert error.message == "Test error"
        assert error.path == "tasks[0].id"
        assert error.error_type == "semantic"
    
    def test_validation_error_defaults(self):
        """Test ValidationError with default values."""
        error = ValidationError(message="Test error")
        assert error.message == "Test error"
        assert error.path == ""
        assert error.error_type == "unknown"


class TestProtocolValidationResult:
    """Test ProtocolValidationResult dataclass."""
    
    def test_result_creation_valid(self):
        """Test creating a valid result."""
        result = ProtocolValidationResult(is_valid=True)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.statistics == {}
    
    def test_result_creation_invalid(self):
        """Test creating an invalid result."""
        result = ProtocolValidationResult(is_valid=False)
        assert result.is_valid is False
    
    def test_add_error(self):
        """Test adding errors to result."""
        result = ProtocolValidationResult(is_valid=True)
        result.add_error("Test error", path="tasks[0]", error_type="schema")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].message == "Test error"
        assert result.errors[0].path == "tasks[0]"
        assert result.errors[0].error_type == "schema"
    
    def test_add_warning(self):
        """Test adding warnings to result."""
        result = ProtocolValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid is True  # Warnings don't invalidate
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"


class TestProtocolValidate:
    """Test Protocol.validate() class method."""

    def test_validate_valid_protocol(self, valid_protocol_yaml):
        """Test validating a valid protocol."""
        result = Protocol.validate(valid_protocol_yaml)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert "task_count" in result.statistics
        assert result.statistics["task_count"] == 2
        assert result.statistics["input_count"] == 3
    
    def test_validate_invalid_yaml_syntax(self, invalid_yaml_syntax):
        """Test validating protocol with invalid YAML syntax."""
        result = Protocol.validate(invalid_yaml_syntax)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(e.error_type == "syntax" for e in result.errors)
    
    def test_validate_missing_schema_version(self, missing_schema_version):
        """Test validating protocol missing required fields (e.g. inputs)."""
        result = Protocol.validate(missing_schema_version)
        assert result.is_valid is False
        assert any("inputs" in e.message or "required" in e.message.lower() for e in result.errors)
        assert any(e.error_type == "schema" for e in result.errors)
    
    def test_validate_invalid_task_reference(self, invalid_task_reference):
        """Test validating protocol with invalid task references."""
        result = Protocol.validate(invalid_task_reference)
        assert result.is_valid is False
        assert any("nonexistent_task" in e.message for e in result.errors)
        assert any(e.error_type == "semantic" for e in result.errors)
    
    def test_validate_circular_dependency(self, circular_dependency):
        """Test validating protocol with circular dependencies."""
        result = Protocol.validate(circular_dependency)
        assert result.is_valid is False
        assert any("circular" in e.message.lower() for e in result.errors)
        assert any(e.error_type == "semantic" for e in result.errors)
    
    def test_validate_invalid_template_expression(self, invalid_template_expression):
        """Test validating protocol with invalid template expressions."""
        result = Protocol.validate(invalid_template_expression)
        # Should catch YAML syntax error first, but if it parses, should catch template error
        assert result.is_valid is False
        # Either syntax error or semantic template error
        assert any(
            e.error_type in ("syntax", "semantic") 
            for e in result.errors
        )
    
    def test_validate_nonexistent_file(self):
        """Test validating a non-existent file (returns invalid result with error, does not raise)."""
        result = Protocol.validate("/nonexistent/path/protocol.yaml")
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Failed" in e.message or "parse" in e.message.lower() for e in result.errors)
    
    def test_validate_collects_all_errors(self, tmp_path):
        """Test that validation collects all errors, not just the first."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  x:
    type: text
    label: X
    required: true
    initial: ''
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items: []
    depends_on: [nonexistent1, nonexistent2]
  - id: task2
    slug: esm2
    action: encode
    request_body:
      items: []
    depends_on: [nonexistent3]
"""
        protocol_file = tmp_path / "multiple_errors.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        assert result.is_valid is False
        # Should have multiple errors (semantic: unknown task IDs)
        assert len(result.errors) > 1


class TestTaskReferenceValidation:
    """Test task reference validation."""
    
    def test_duplicate_task_ids(self, tmp_path):
        """Test detection of duplicate task IDs."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: task1
    slug: esmfold
    action: predict
  - id: task1
    slug: esm2-8m
    action: encode
"""
        protocol_file = tmp_path / "duplicate_ids.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        assert result.is_valid is False
        assert any("duplicate" in e.message.lower() for e in result.errors)
    
    def test_invalid_from_reference(self, tmp_path):
        """Test invalid 'from' reference."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: task1
    slug: esmfold
    action: predict
  - id: task2
    type: gather
    from: nonexistent
"""
        protocol_file = tmp_path / "invalid_from.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        assert result.is_valid is False
        assert any("nonexistent" in e.message for e in result.errors)
    
    def test_invalid_foreach_reference(self, tmp_path):
        """Test invalid 'foreach' reference (simple string, not template)."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: task1
    slug: esmfold
    action: predict
  - id: task2
    slug: esm2-8m
    action: encode
    foreach: nonexistent
"""
        protocol_file = tmp_path / "invalid_foreach.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        assert result.is_valid is False
        assert any("nonexistent" in e.message for e in result.errors)


class TestTemplateExpressionValidation:
    """Test template expression validation."""
    
    def test_valid_template_expressions(self, tmp_path):
        """Test that valid template expressions pass validation."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items:
        - sequence: ${{ sequence }}
      params:
        count: ${{ n_samples // 2 }}
"""
        protocol_file = tmp_path / "valid_templates.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        # Should not have template expression errors
        template_errors = [
            e for e in result.errors 
            if "template" in e.message.lower() or "expression" in e.message.lower()
        ]
        assert len(template_errors) == 0
    
    def test_unbalanced_braces(self, tmp_path):
        """Test detection of unbalanced braces in template expressions."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items:
        - sequence: ${{ invalid { expression }}
"""
        protocol_file = tmp_path / "unbalanced.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        assert result.is_valid is False
        assert any("brace" in e.message.lower() for e in result.errors)
    
    def test_malformed_template(self, tmp_path):
        """Test detection of malformed template expressions."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items:
        - sequence: ${{ invalid expression
"""
        protocol_file = tmp_path / "malformed.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        # Should catch either YAML syntax error or template error
        assert result.is_valid is False


class TestStatisticsCollection:
    """Test statistics collection."""
    
    def test_statistics_for_valid_protocol(self, valid_protocol_yaml):
        """Test statistics collection for valid protocol."""
        result = Protocol.validate(valid_protocol_yaml)
        stats = result.statistics
        
        assert "protocol_name" in stats
        assert stats["protocol_name"] == "Test Protocol"
        assert "task_count" in stats
        assert stats["task_count"] == 2
        assert "input_count" in stats
        assert stats["input_count"] == 3
        assert "model_task_count" in stats
        assert "gather_task_count" in stats
    
    def test_statistics_task_types(self, tmp_path):
        """Test statistics distinguish between model and gather tasks."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: model1
    slug: esmfold
    action: predict
  - id: model2
    slug: esm2-8m
    action: encode
  - id: gather1
    type: gather
    from: model1
"""
        protocol_file = tmp_path / "task_types.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        stats = result.statistics
        
        assert stats["task_count"] == 3
        assert stats["model_task_count"] == 2
        assert stats["gather_task_count"] == 1
    
    def test_statistics_output_rules(self, tmp_path):
        """Test statistics include output rule count."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks:
  - id: task1
    slug: esmfold
    action: predict
outputs:
  - from: task1
    order_by: pdb
    limit: 10
  - from: task1
    order_by: mean_plddt
    limit: 5
"""
        protocol_file = tmp_path / "outputs.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        stats = result.statistics
        
        assert "output_rule_count" in stats
        assert stats["output_rule_count"] == 2


class TestProtocolInitBackwardCompatibility:
    """Test that Protocol.__init__ still works with validation."""

    def test_init_valid_protocol(self, valid_protocol_yaml):
        """Test Protocol can be instantiated with valid protocol."""
        protocol = Protocol(valid_protocol_yaml)
        assert protocol.yaml_path == valid_protocol_yaml
        assert protocol.data is not None
        assert "name" in protocol.data
    
    def test_init_invalid_protocol_raises(self, missing_schema_version):
        """Test Protocol.__init__ raises on invalid protocol."""
        with pytest.raises(ValueError):
            Protocol(missing_schema_version)
    
    def test_init_invalid_yaml_raises(self, invalid_yaml_syntax):
        """Test Protocol.__init__ raises on invalid YAML."""
        with pytest.raises(ValueError):
            Protocol(invalid_yaml_syntax)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_protocol(self, tmp_path):
        """Test validation of empty protocol."""
        protocol_file = tmp_path / "empty.yaml"
        protocol_file.write_text("{}")
        
        result = Protocol.validate(str(protocol_file))
        assert result.is_valid is False
        # Should have schema errors for missing required fields
        assert any("required" in e.message.lower() for e in result.errors)
    
    def test_protocol_with_no_tasks(self, tmp_path):
        """Test validation of protocol with no tasks."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
tasks: []
"""
        protocol_file = tmp_path / "no_tasks.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        assert result.is_valid is False
        # Should have schema error for empty tasks array
    
    def test_protocol_with_no_inputs(self, tmp_path):
        """Test validation of protocol with no inputs."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs: {}
tasks:
  - id: task1
    slug: esmfold
    action: predict
"""
        protocol_file = tmp_path / "no_inputs.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        # Empty inputs should be valid
        assert result.statistics["input_count"] == 0
    
    def test_nested_template_expressions(self, tmp_path):
        """Test validation with nested structures containing template expressions."""
        protocol_content = """
name: Test Protocol
schema_version: 1
inputs:
  n_samples: 10
  items: ["item1", "item2"]
tasks:
  - id: task1
    slug: esmfold
    action: predict
    request_body:
      items: ${{ items }}
      params:
        count: ${{ n_samples }}
        nested:
          value: ${{ n_samples * 2 }}
"""
        protocol_file = tmp_path / "nested.yaml"
        protocol_file.write_text(protocol_content)
        
        result = Protocol.validate(str(protocol_file))
        # Should validate successfully (template expressions are valid)
        template_errors = [
            e for e in result.errors 
            if "template" in e.message.lower() or "expression" in e.message.lower()
        ]
        assert len(template_errors) == 0


class TestCLIValidate:
    """Test CLI protocol validate command."""
    
    def test_cli_validate_valid_protocol(self, valid_protocol_yaml):
        """Test CLI validate command with valid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["protocol", "validate", valid_protocol_yaml])
        
        assert result.exit_code == 0
        assert "Validation Successful" in result.output or "✓" in result.output
        assert "valid" in result.output.lower()
    
    def test_cli_validate_invalid_protocol(self, missing_schema_version):
        """Test CLI validate command with invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["protocol", "validate", missing_schema_version])
        
        assert result.exit_code == 1
        assert "Validation Failed" in result.output or "✗" in result.output
        assert "error" in result.output.lower()
    
    def test_cli_validate_json_output(self, valid_protocol_yaml):
        """Test CLI validate command with JSON output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["protocol", "validate", "--json", valid_protocol_yaml])
        
        assert result.exit_code == 0
        # Should be valid JSON
        output_data = json.loads(result.output)
        assert "valid" in output_data
        assert output_data["valid"] is True
        assert "errors" in output_data
        assert "warnings" in output_data
        assert "statistics" in output_data
    
    def test_cli_validate_json_output_invalid(self, missing_schema_version):
        """Test CLI validate command with JSON output for invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["protocol", "validate", "--json", missing_schema_version])
        
        assert result.exit_code == 1
        # Output may be JSON only (strip any leading/trailing whitespace or Rich artifacts)
        output_str = result.output.strip()
        output_data = json.loads(output_str)
        assert "valid" in output_data
        assert output_data["valid"] is False
        assert "errors" in output_data
        assert len(output_data["errors"]) > 0
    
    def test_cli_validate_nonexistent_file(self):
        """Test CLI validate command with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["protocol", "validate", "/nonexistent/path.yaml"])
        
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "Error" in result.output or "no such file" in result.output.lower()
    
    def test_cli_validate_shows_statistics(self, valid_protocol_yaml):
        """Test CLI validate command shows statistics for valid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["protocol", "validate", valid_protocol_yaml])
        
        assert result.exit_code == 0
        # Should show statistics
        assert "task" in result.output.lower() or "input" in result.output.lower()
    
    def test_cli_validate_shows_errors_table(self, invalid_task_reference):
        """Test CLI validate command shows errors in a table."""
        runner = CliRunner()
        result = runner.invoke(cli, ["protocol", "validate", invalid_task_reference])
        
        assert result.exit_code == 1
        # Should show error details
        assert "nonexistent_task" in result.output or "error" in result.output.lower()

