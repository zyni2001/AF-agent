import re
from typing import Dict

from . import my_a2a
from .llm_provider import build_litellm_kwargs, get_api_keys_from_env, use_vertex_ai


def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """the target str contains tags in the format of <tag_name> ... </tag_name>, parse them out and return a dict"""

    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}
