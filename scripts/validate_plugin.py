#!/usr/bin/env python3
"""Validate Claude Code plugin structure and components.

This script validates:
- Plugin metadata (plugin.json)
- Command files (.md in commands/)
- Agent files (.md in agents/)
- Skill files (SKILL.md in skills/)
- Hook configuration (hooks.json)
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any

import yaml


# Constants
MIN_FRONTMATTER_PARTS = 3  # Frontmatter requires: start ---, content, end ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PluginValidator:
    """Validator for Claude Code plugin structure."""

    def __init__(self, plugin_dir: Path) -> None:
        """Initialize validator.

        Args:
            plugin_dir: Root directory of the plugin
        """
        self.plugin_dir = plugin_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating plugin at: %s", self.plugin_dir)

        # Run validation checks
        self._validate_plugin_json()
        self._validate_commands()
        self._validate_agents()
        self._validate_skills()
        self._validate_hooks()

        # Report results
        if self.warnings:
            logger.warning("Found %d warning(s):", len(self.warnings))
            for warning in self.warnings:
                logger.warning("  - %s", warning)

        if self.errors:
            logger.error("Found %d error(s):", len(self.errors))
            for error in self.errors:
                logger.error("  - %s", error)
            return False

        logger.info("✓ Plugin validation passed!")
        return True

    def _validate_plugin_json(self) -> None:
        """Validate plugin.json structure."""
        plugin_json_path = self.plugin_dir / ".claude-plugin" / "plugin.json"

        if not plugin_json_path.exists():
            self.errors.append("plugin.json not found at .claude-plugin/plugin.json")
            return

        try:
            with plugin_json_path.open() as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in plugin.json: {e}")
            return

        # Check required fields
        required_fields = ["name", "version", "description"]
        for field in required_fields:
            if field not in data:
                self.errors.append(f"Missing required field in plugin.json: {field}")

        # Validate optional fields if present
        if "agents" in data:
            self._validate_agent_paths(data["agents"])

        if "skills" in data:
            self._validate_skill_paths(data["skills"])

        if "lspServers" in data:
            self._validate_lsp_servers(data["lspServers"])

        logger.info("✓ plugin.json structure is valid")

    def _validate_agent_paths(self, agents: list[str]) -> None:
        """Validate agent file paths exist.

        Args:
            agents: List of agent file paths from plugin.json
        """
        for agent_path in agents:
            full_path = self.plugin_dir / agent_path
            if not full_path.exists():
                self.errors.append(f"Agent file not found: {agent_path}")

    def _validate_skill_paths(self, skills: list[str]) -> None:
        """Validate skill directory paths exist.

        Args:
            skills: List of skill directory paths from plugin.json
        """
        for skill_path in skills:
            full_path = self.plugin_dir / skill_path
            if not full_path.exists():
                self.errors.append(f"Skill directory not found: {skill_path}")

    def _validate_lsp_servers(
        self, lsp_servers: dict[str, dict[str, Any]] | list[dict[str, Any]]
    ) -> None:
        """Validate LSP server configurations.

        Args:
            lsp_servers: LSP server configs from plugin.json
                (dict or list for backward compatibility)
        """
        # Handle both dict (new format) and list (old format) for backward compatibility
        if isinstance(lsp_servers, dict):
            servers = lsp_servers.items()
            for server_name, server_config in servers:
                if "command" not in server_config:
                    self.errors.append(f"LSP server '{server_name}' missing 'command'")
        else:
            # Old list format
            for server in lsp_servers:
                if "name" not in server:
                    self.errors.append("LSP server missing 'name' field")
                if "command" not in server:
                    server_name = server.get("name", "unknown")
                    self.errors.append(f"LSP server '{server_name}' missing 'command'")

    def _validate_commands(self) -> None:
        """Validate command files."""
        commands_dir = self.plugin_dir / "commands"

        if not commands_dir.exists():
            self.warnings.append("No commands/ directory found")
            return

        command_files = list(commands_dir.glob("*.md"))
        if not command_files:
            self.warnings.append("No command files found in commands/")
            return

        for cmd_file in command_files:
            self._validate_command_file(cmd_file)

        logger.info("✓ Found %d command file(s)", len(command_files))

    def _validate_command_file(self, cmd_file: Path) -> None:
        """Validate individual command file.

        Args:
            cmd_file: Path to command markdown file
        """
        try:
            content = cmd_file.read_text()
            if not content.strip():
                self.warnings.append(f"Command file is empty: {cmd_file.name}")
        except Exception as e:
            self.errors.append(f"Error reading command file {cmd_file.name}: {e}")

    def _validate_agents(self) -> None:
        """Validate agent files."""
        agents_dir = self.plugin_dir / "agents"

        if not agents_dir.exists():
            self.warnings.append("No agents/ directory found")
            return

        agent_files = list(agents_dir.glob("*.md"))
        if not agent_files:
            self.warnings.append("No agent files found in agents/")
            return

        for agent_file in agent_files:
            self._validate_agent_file(agent_file)

        logger.info("✓ Found %d agent file(s)", len(agent_files))

    def _validate_agent_file(self, agent_file: Path) -> None:
        """Validate individual agent file.

        Args:
            agent_file: Path to agent markdown file
        """
        try:
            content = agent_file.read_text()

            # Check for YAML frontmatter
            if not content.startswith("---"):
                self.errors.append(f"Agent file missing YAML frontmatter: {agent_file.name}")
                return

            # Extract and parse frontmatter
            parts = content.split("---", 2)
            if len(parts) < MIN_FRONTMATTER_PARTS:
                self.errors.append(f"Invalid YAML frontmatter in: {agent_file.name}")
                return

            try:
                frontmatter = yaml.safe_load(parts[1])
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {agent_file.name}: {e}")
                return

            # Check required frontmatter fields
            required_fields = ["name", "description", "tools", "model"]
            for field in required_fields:
                if field not in frontmatter:
                    self.errors.append(f"Agent {agent_file.name} missing required field: {field}")

        except Exception as e:
            self.errors.append(f"Error reading agent file {agent_file.name}: {e}")

    def _validate_skills(self) -> None:
        """Validate skill directories."""
        skills_dir = self.plugin_dir / "skills"

        if not skills_dir.exists():
            self.warnings.append("No skills/ directory found")
            return

        skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]
        if not skill_dirs:
            self.warnings.append("No skill directories found in skills/")
            return

        for skill_dir in skill_dirs:
            self._validate_skill_dir(skill_dir)

        logger.info("✓ Found %d skill director(ies)", len(skill_dirs))

    def _validate_skill_dir(self, skill_dir: Path) -> None:
        """Validate individual skill directory.

        Args:
            skill_dir: Path to skill directory
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            self.errors.append(f"Skill directory missing SKILL.md: {skill_dir.name}")

    def _validate_hooks(self) -> None:
        """Validate hooks configuration."""
        hooks_json = self.plugin_dir / "hooks" / "hooks.json"

        if not hooks_json.exists():
            self.warnings.append("No hooks.json found")
            return

        try:
            with hooks_json.open() as f:
                data = json.load(f)

            if "hooks" not in data:
                self.errors.append("hooks.json missing 'hooks' field")
                return

            logger.info("✓ hooks.json is valid")

        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in hooks.json: {e}")


def main() -> int:
    """Run plugin validation.

    Returns:
        0 if validation passes, 1 otherwise
    """
    # Default to current directory
    plugin_dir = Path.cwd()

    # Allow override via command line
    if len(sys.argv) > 1:
        plugin_dir = Path(sys.argv[1])

    if not plugin_dir.exists():
        logger.error("Plugin directory not found: %s", plugin_dir)
        return 1

    validator = PluginValidator(plugin_dir)
    success = validator.validate()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
