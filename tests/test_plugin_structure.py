"""Test plugin structure and metadata."""

import json
from pathlib import Path

import pytest
import yaml


# Get plugin root directory
PLUGIN_DIR = Path(__file__).parent.parent


class TestPluginStructure:
    """Test plugin directory structure and files."""

    def test_plugin_json_exists(self) -> None:
        """Check that plugin.json exists."""
        plugin_json = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
        assert plugin_json.exists(), "plugin.json not found"

    def test_plugin_json_valid(self) -> None:
        """Validate plugin.json structure."""
        plugin_json = PLUGIN_DIR / ".claude-plugin" / "plugin.json"

        with plugin_json.open() as f:
            data = json.load(f)

        # Check required fields
        required_fields = ["name", "version", "description"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Check field types
        assert isinstance(data["name"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["description"], str)

    def test_plugin_json_valid_agents(self) -> None:
        """Validate agents list in plugin.json."""
        plugin_json = PLUGIN_DIR / ".claude-plugin" / "plugin.json"

        with plugin_json.open() as f:
            data = json.load(f)

        if "agents" in data:
            assert isinstance(data["agents"], list)
            for agent_path in data["agents"]:
                agent_file = PLUGIN_DIR / agent_path
                assert agent_file.exists(), f"Agent file not found: {agent_path}"

    def test_plugin_json_valid_skills(self) -> None:
        """Validate skills list in plugin.json."""
        plugin_json = PLUGIN_DIR / ".claude-plugin" / "plugin.json"

        with plugin_json.open() as f:
            data = json.load(f)

        if "skills" in data:
            assert isinstance(data["skills"], list)
            for skill_path in data["skills"]:
                skill_dir = PLUGIN_DIR / skill_path
                assert skill_dir.exists(), f"Skill path not found: {skill_path}"

    def test_commands_directory_exists(self) -> None:
        """Check that commands directory exists."""
        commands_dir = PLUGIN_DIR / "commands"
        assert commands_dir.exists(), "commands/ directory not found"
        assert commands_dir.is_dir(), "commands/ is not a directory"

    def test_commands_have_content(self) -> None:
        """Check that command files have content."""
        commands_dir = PLUGIN_DIR / "commands"
        command_files = list(commands_dir.glob("*.md"))

        assert len(command_files) > 0, "No command files found"

        for cmd_file in command_files:
            content = cmd_file.read_text()
            assert content.strip(), f"Command file is empty: {cmd_file.name}"

    def test_agents_directory_exists(self) -> None:
        """Check that agents directory exists."""
        agents_dir = PLUGIN_DIR / "agents"
        assert agents_dir.exists(), "agents/ directory not found"
        assert agents_dir.is_dir(), "agents/ is not a directory"

    def test_agents_have_valid_frontmatter(self) -> None:
        """Check that agent files have valid YAML frontmatter."""
        agents_dir = PLUGIN_DIR / "agents"
        agent_files = list(agents_dir.glob("*.md"))

        assert len(agent_files) > 0, "No agent files found"

        for agent_file in agent_files:
            content = agent_file.read_text()

            # Check for frontmatter
            assert content.startswith("---"), f"Agent {agent_file.name} missing frontmatter"

            # Parse frontmatter
            parts = content.split("---", 2)
            assert len(parts) >= 3, f"Invalid frontmatter in {agent_file.name}"

            frontmatter = yaml.safe_load(parts[1])

            # Check required fields
            required_fields = ["name", "description", "tools", "model"]
            for field in required_fields:
                assert field in frontmatter, (
                    f"Agent {agent_file.name} missing required field: {field}"
                )

            # Validate types
            assert isinstance(frontmatter["name"], str)
            assert isinstance(frontmatter["description"], str)
            assert isinstance(frontmatter["tools"], list)
            assert isinstance(frontmatter["model"], str)

    def test_skills_directory_exists(self) -> None:
        """Check that skills directory exists."""
        skills_dir = PLUGIN_DIR / "skills"
        assert skills_dir.exists(), "skills/ directory not found"
        assert skills_dir.is_dir(), "skills/ is not a directory"

    def test_skills_have_skill_md(self) -> None:
        """Check that skill directories have SKILL.md."""
        skills_dir = PLUGIN_DIR / "skills"
        skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]

        assert len(skill_dirs) > 0, "No skill directories found"

        for skill_dir in skill_dirs:
            skill_md = skill_dir / "SKILL.md"
            assert skill_md.exists(), f"Skill directory {skill_dir.name} missing SKILL.md"

            content = skill_md.read_text()
            assert content.strip(), f"SKILL.md is empty in {skill_dir.name}"

    def test_hooks_json_exists(self) -> None:
        """Check that hooks.json exists (optional)."""
        hooks_json = PLUGIN_DIR / "hooks" / "hooks.json"
        # This is optional, so just check if it exists and is valid JSON if present
        if hooks_json.exists():
            with hooks_json.open() as f:
                data = json.load(f)
            assert "hooks" in data, "hooks.json missing 'hooks' field"

    def test_readme_exists(self) -> None:
        """Check that README.md exists."""
        readme = PLUGIN_DIR / "README.md"
        assert readme.exists(), "README.md not found"
        assert readme.read_text().strip(), "README.md is empty"


class TestCommandFiles:
    """Test individual command files."""

    @pytest.fixture
    def command_files(self) -> list[Path]:
        """Get all command files."""
        commands_dir = PLUGIN_DIR / "commands"
        return list(commands_dir.glob("*.md"))

    def test_command_files_are_markdown(self, command_files: list[Path]) -> None:
        """Check that all command files are markdown."""
        for cmd_file in command_files:
            assert cmd_file.suffix == ".md", f"{cmd_file.name} is not a markdown file"

    def test_command_files_have_title(self, command_files: list[Path]) -> None:
        """Check that command files have a title (# heading)."""
        for cmd_file in command_files:
            content = cmd_file.read_text()
            lines = content.split("\n")

            # Skip YAML frontmatter if present
            start_idx = 0
            if lines and lines[0].strip() == "---":
                # Find end of frontmatter
                for i, line in enumerate(lines[1:], start=1):
                    if line.strip() == "---":
                        start_idx = i + 1
                        break

            # Find first non-empty line after frontmatter
            first_line = next(
                (line for line in lines[start_idx:] if line.strip()), ""
            )

            assert first_line.startswith("#"), (
                f"Command {cmd_file.name} missing title (should start with #)"
            )


class TestAgentFiles:
    """Test individual agent files."""

    @pytest.fixture
    def agent_files(self) -> list[Path]:
        """Get all agent files."""
        agents_dir = PLUGIN_DIR / "agents"
        return list(agents_dir.glob("*.md"))

    def test_agent_names_match_filenames(self, agent_files: list[Path]) -> None:
        """Check that agent names in frontmatter match filenames."""
        for agent_file in agent_files:
            content = agent_file.read_text()
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])

            # Get filename without extension
            filename_base = agent_file.stem

            # Agent name should match filename
            agent_name = frontmatter["name"]
            assert agent_name == filename_base, (
                f"Agent name '{agent_name}' doesn't match filename '{filename_base}'"
            )

    def test_agent_models_are_valid(self, agent_files: list[Path]) -> None:
        """Check that agent models are valid."""
        valid_models = ["sonnet", "opus", "haiku"]

        for agent_file in agent_files:
            content = agent_file.read_text()
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])

            model = frontmatter["model"]
            assert model in valid_models, f"Agent {agent_file.name} has invalid model: {model}"


class TestPluginMetadata:
    """Test plugin metadata consistency."""

    def test_version_format(self) -> None:
        """Check that version follows semantic versioning."""
        plugin_json = PLUGIN_DIR / ".claude-plugin" / "plugin.json"

        with plugin_json.open() as f:
            data = json.load(f)

        version = data["version"]
        parts = version.split(".")

        assert len(parts) >= 2, f"Version {version} doesn't follow semver"
        assert all(part.isdigit() for part in parts), f"Version {version} has non-numeric parts"

    def test_keywords_exist(self) -> None:
        """Check that keywords are defined (optional but recommended)."""
        plugin_json = PLUGIN_DIR / ".claude-plugin" / "plugin.json"

        with plugin_json.open() as f:
            data = json.load(f)

        # Keywords are optional but recommended
        if "keywords" in data:
            assert isinstance(data["keywords"], list)
            assert len(data["keywords"]) > 0, "Keywords list is empty"
