# __init__.py
# 暂时注释掉同步API相关的导入，因为它们还在使用旧的代理模块
# 等后续统一更新代理实现后再启用
# from .api import PoeApi
from .async_api import AsyncPoeApi
# from .example import PoeExample  # 依赖于PoeApi，暂时注释

from .llm import LLM_PACKAGE
if LLM_PACKAGE:
    from .llm import PoeServer

# 注意：
# 1. 移除了原来的代理检测和环境变量设置
# 2. 暂时只保留AsyncPoeApi的导入，因为已完成新代理机制的迁移
# 3. 同步API(PoeApi)及其相关模块待更新后再启用