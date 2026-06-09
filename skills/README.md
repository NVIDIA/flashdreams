# Agent Skills

Project-authored [Agent Skills](https://agentskills.io/specification) live in
the canonical root `skills/` directory. They are **opt-in**: FlashDreams
does not assume that every agent client automatically loads repository
skills, so developers can still control which local skills their tools use.

Opt in by symlinking this directory into whichever tool you use. From the repo root:

```bash
# Cursor
mkdir -p .cursor && ln -s ../skills .cursor/skills

# Claude Code
mkdir -p .claude && ln -s ../skills .claude/skills

# Other / general agents
mkdir -p .agents && ln -s ../skills .agents/skills
```

`.cursor/` / `.claude/` / `.agents/` are already gitignored (add entries if they aren't), so the symlinks stay local. If you want to be selective, symlink individual skill directories instead of the whole folder:

```bash
ln -s ../../skills/<skill-name> <.cursor|.claude|.agents>/skills/<skill-name>
```

## Layout

Each skill is a directory containing a `SKILL.md` with YAML frontmatter:

```
skills/
├── README.md
└── <skill-name>/
    ├── SKILL.md          # required: name, description, body
    ├── reference.md      # optional: deeper docs, linked from SKILL.md
    └── scripts/          # optional: executable helpers
```

The `SKILL.md` format follows the vendor-neutral specification [agentskills.io](https://agentskills.io/specification) and is compatible with claude, cursor, and other agents: YAML frontmatter with `name` (lowercase-hyphenated, ≤64 chars) and `description` (specific, third person, includes both *what* and *when*), followed by markdown instructions. Keep the body under ~500 lines and push long-form reference into sibling files.

## Authoring

When adding a skill, prefer codifying conventions already visible in the codebase over re-deriving them. The goal is to reduce drift between human-written code and agent-written code, not to invent new style.
