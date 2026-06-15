from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SKILLS_DIR = Path(__file__).resolve().parent / "skills"
TEXT_RESOURCE_EXTENSIONS = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".py", ".js", ".ts", ".sh"}
CONVENTIONAL_RESOURCE_DIRS = {"references", "assets", "scripts"}

# 这是一个资源文件的类，用于表示技能中的资源文件
@dataclass
class SkillResource:
    path: str   # 相对路径
    kind: str   # 资源类型
    source: str   # 发现来源
    size_bytes: int   # 文件大小
    loadable: bool    # 是否允许读取

# 这是一个技能索引的类，用于表示一个完整的Skill
@dataclass
class SkillIndex:
    name: str   # 技能名称
    description: str   # 技能描述
    folder_name: str   # 目录名
    entry_file: str = "SKILL.md"   # 入口文件
    references: list[str] = field(default_factory=list)            # 引用资源
    resources: list[SkillResource] = field(default_factory=list)   # 扫描出的资源
    metadata: dict[str, Any] = field(default_factory=dict)         # 额外元数据

# 用于解析Markdown开头的YAML风格的frontmatter元数据，把解析结果整理成一个dict[str, Any]，读取SkillIndex的元数据
def parse_frontmatter(content: str) -> dict[str, Any]:
    if not content.startswith("---\n"):
        return {}

    closing = content.find("\n---\n", 4)
    if closing == -1:
        return {}

    header = content[4:closing]
    metadata: dict[str, Any] = {}
    lines = header.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            index += 1
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value in {">", "|"}:
            index += 1
            buffer: list[str] = []
            while index < len(lines):
                continuation = lines[index]
                if continuation.startswith(" ") or continuation.startswith("\t"):
                    buffer.append(continuation.strip())
                    index += 1
                    continue
                break
            metadata[key] = " ".join(buffer).strip()
            continue

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            metadata[key] = [part.strip().strip("\"'") for part in inner.split(",") if part.strip()]
        else:
            metadata[key] = value.strip("\"'")

        index += 1

    return metadata

# 对传入的resource_path进行路径清洗和校验，返回一个相对路径
def _normalize_resource_path(resource_path: str) -> str | None:
    if not isinstance(resource_path, str):
        return None

    clean_path = resource_path.strip().replace("\\", "/")
    if not clean_path or clean_path.startswith("#"):
        return None
    if os.path.isabs(clean_path):
        return None

    normalized = os.path.normpath(clean_path).replace("\\", "/")
    if normalized in {".", ".."} or normalized.startswith("../"):
        return None
    return normalized

# 把一个skill目录下的资源相对路径，解析成最终的绝对路径，同时做安全校验，如果路径有越界风险或格式非法，就返回None
def resolve_skill_resource_path(skill_dir: str | Path, resource_path: str) -> str | None:
    normalized = _normalize_resource_path(resource_path)
    if not normalized:
        return None

    base = os.path.realpath(str(skill_dir))
    target = os.path.realpath(os.path.join(str(skill_dir), normalized))
    try:
        if os.path.commonpath([base, target]) != base:
            return None
    except ValueError:
        return None
    return target

# 在Markdown 文本里找出所有链接目标，并筛选出其中安全的相对文件路径 
def _extract_markdown_links(content: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", content):
        raw_link = match.group(1).strip().split("#", 1)[0].strip().strip("\"'")
        normalized = _normalize_resource_path(raw_link)
        if normalized:
            links.append(normalized)
    return links

# 把一个目标文件路径转换成“相对于技能目录的标准相对路径表示”
def _relative_path(skill_dir: str | Path, target_path: str | Path) -> str:
    return os.path.relpath(os.path.realpath(str(target_path)), os.path.realpath(str(skill_dir))).replace(os.sep, "/")

# 根据资源的相对路径relative_path，判断它属于哪一类资源，并标记它是否应该按文本文件处理
def _classify_resource(relative_path: str) -> tuple[str, bool]:
    lower_path = relative_path.lower()
    _, ext = os.path.splitext(lower_path)   # 取出文件扩展名

    if ext in {".md", ".markdown"}:
        kind = "markdown"
    elif lower_path.startswith("references/"):
        kind = "reference"
    elif lower_path.startswith("assets/"):
        kind = "asset"
    elif lower_path.startswith("scripts/"):
        kind = "script"
    else:
        kind = "other"

    return kind, ext in TEXT_RESOURCE_EXTENSIONS or kind == "markdown"

# 扫描一个skill目录相关的资源文件，把找到的资源整理成SkillResource对象列表，最终返回一个按路径排序的资源清单
def discover_skill_resources(skill_dir: str | Path, main_content: str, frontmatter: dict[str, Any]) -> list[SkillResource]:
    resources: dict[str, SkillResource] = {}

    def add_resource(relative_path: str, source: str) -> None:
        normalized = _normalize_resource_path(relative_path)
        if not normalized:
            return

        absolute_path = resolve_skill_resource_path(skill_dir, normalized)
        if not absolute_path or not os.path.isfile(absolute_path):
            return

        canonical_rel = _relative_path(skill_dir, absolute_path)
        if canonical_rel in resources:
            return

        kind, loadable = _classify_resource(canonical_rel)
        resources[canonical_rel] = SkillResource(
            path=canonical_rel,   # 相对路径
            kind=kind,            # 资源类型
            source=source,        # 资源发现来源
            size_bytes=os.path.getsize(absolute_path),   # 文件大小
            loadable=loadable,                           # 是否适合作为文本资源读取
        )

    references = frontmatter.get("references", [])
    if isinstance(references, str):
        references = [item.strip() for item in references.split(",") if item.strip()]
    if isinstance(references, list):
        for reference in references:
            if isinstance(reference, str):
                add_resource(reference, "frontmatter")

    for link in _extract_markdown_links(main_content):
        add_resource(link, "markdown_link")

    for dirname in CONVENTIONAL_RESOURCE_DIRS:
        root = os.path.join(str(skill_dir), dirname)
        if not os.path.isdir(root):
            continue
        for current_root, _, files in os.walk(root):
            for filename in files:
                add_resource(_relative_path(skill_dir, os.path.join(current_root, filename)), "conventional_dir")

    return sorted(resources.values(), key=lambda item: item.path)

# 扫描SKILLS_DIR目录下所有可用的skill，读取每个skill的SKILL.md，建立整个skills目录的索引清单
def scan_skill_index() -> list[SkillIndex]:
    skill_indices: list[SkillIndex] = []
    if not SKILLS_DIR.exists():
        return skill_indices

    for item in sorted(SKILLS_DIR.iterdir()):
        if not item.is_dir():
            continue

        entry_file = "SKILL.md"
        entry_path = item / entry_file
        if not entry_path.exists():
            continue

        content = entry_path.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(content)   # 解析顶部frontmatter
        metadata = {key: value for key, value in frontmatter.items() if key not in {"name", "description", "references"}}
        raw_refs = frontmatter.get("references", [])
        if isinstance(raw_refs, str):
            references = [part.strip() for part in raw_refs.split(",") if part.strip()]
        elif isinstance(raw_refs, list):
            references = [part for part in raw_refs if isinstance(part, str)]
        else:
            references = []

        # 将skill的信息封装成一个SkillIndex对象
        skill_indices.append(
            SkillIndex(
                name=str(frontmatter.get("name", item.name)).strip() or item.name,
                description=str(frontmatter.get("description", f"处理 {item.name} 相关任务。")).strip(),
                folder_name=item.name,   
                entry_file=entry_file,
                references=references,
                resources=discover_skill_resources(item, content, frontmatter),
                metadata=metadata,
            )
        )

    return skill_indices

# 在当前所有已扫描的skill中，根据skill名称或文件夹名称查找对应的SkillIndex对象
def _find_skill(skill_name: str) -> SkillIndex | None:
    for skill in scan_skill_index():
        if skill.name == skill_name or skill.folder_name == skill_name:
            return skill
    return None

# 把当前扫描到的所有skill索引信息，整理成一段适合展示的文本
def get_skill_index_text(max_description_chars: int = 400) -> str:
    skill_indices = scan_skill_index()
    if not skill_indices:
        return "当前没有可用的外部 skill。"

    lines: list[str] = []
    for skill in skill_indices:
        description = " ".join(skill.description.split())
        if len(description) > max_description_chars:
            description = description[:max_description_chars].rstrip() + "..."
        lines.append(f"- Skill: {skill.name}")
        lines.append(f"  Description: {description}")
        if skill.resources:
            resource_names = ", ".join(resource.path for resource in skill.resources[:5])
            if len(skill.resources) > 5:
                resource_names += ", ..."
            lines.append(f"  Resources: {resource_names}")
    return "\n".join(lines)


def list_skills_text() -> str:
    return get_skill_index_text()

# 加载指定skill的SKILL.md文件内容
def load_skill_content(skill_name: str) -> str:
    skill = _find_skill(skill_name)
    if not skill:
        available = ", ".join(item.name for item in scan_skill_index()) or "无"
        return f"错误：未找到名为 '{skill_name}' 的 skill。可用 skill：{available}"

    path = SKILLS_DIR / skill.folder_name / skill.entry_file
    return path.read_text(encoding="utf-8")

# 列出指定skill的所有附带资源
def list_skill_resources_content(skill_name: str) -> str:
    skill = _find_skill(skill_name)
    if not skill:
        return f"错误：未找到 skill '{skill_name}'。"
    if not skill.resources:
        return "该 skill 没有附带资源。"

    lines = ["Bundled Resources:"]
    for resource in skill.resources:
        loadable_text = "loadable" if resource.loadable else "not-loadable"
        lines.append(
            f"- {resource.path} ({resource.kind}; {resource.size_bytes} bytes; source={resource.source}; {loadable_text})"
        )
    lines.append("如需读取文本资源，请调用 load_skill_resource(skill_name, resource_path)。")
    return "\n".join(lines)

# 加载指定skill的指定资源内容
def load_skill_resource_content(skill_name: str, resource_path: str) -> str:
    skill = _find_skill(skill_name)
    if not skill:
        return f"错误：未找到 skill '{skill_name}'。"

    skill_dir = SKILLS_DIR / skill.folder_name
    absolute_path = resolve_skill_resource_path(skill_dir, resource_path)
    if not absolute_path:
        return "错误：资源路径不安全，必须使用 skill 目录内的相对路径。"
    if not os.path.exists(absolute_path) or os.path.isdir(absolute_path):
        return f"错误：资源 '{resource_path}' 不存在。"

    _, loadable = _classify_resource(_relative_path(skill_dir, absolute_path))
    if not loadable:
        return f"错误：资源 '{resource_path}' 不是可读取的文本资源。"

    return Path(absolute_path).read_text(encoding="utf-8", errors="replace")
