"""MkDocs hooks for the macbot docs site.

The Markdown files are written to read well on GitHub too, so two things are adjusted for the site:

- Each page starts with a link to its other language ("[简体中文](x.zh.md)" or "[English](x.md)"). The site has a
  language switcher in the header, so the hook drops that line.
- Links to repository files outside docs/ ("../configs/so101.yaml") work on GitHub but not on the site. The hook
  points them at the file on GitHub.
"""
import re

_LANG_LINK = re.compile(r"^\[(简体中文|中文|English)\]\([^)]+\)[ \t]*\n", re.MULTILINE)
_REPO_LINK = re.compile(r"\]\(\.\./([^)#\s]+)((?:#[^)\s]*)?)\)")


def on_page_markdown(markdown, page, config, files):
    markdown = _LANG_LINK.sub("", markdown, count=1)
    repo = config.get("repo_url", "").rstrip("/")
    if repo:
        def to_github(match):
            path, anchor = match.group(1), match.group(2)
            kind = "tree" if path.endswith("/") else "blob"
            return f"]({repo}/{kind}/main/{path}{anchor})"

        markdown = _REPO_LINK.sub(to_github, markdown)
    return markdown
