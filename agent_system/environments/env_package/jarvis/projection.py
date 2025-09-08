# agent_system/environments/env_package/jarvis/projection.py

from typing import Any, Dict

class JarvisProjection:
    """
    一个简单的投射器，用于Jarvis环境。
    由于Jarvis环境的观测和动作都已经是自然语言字符串，
    这个类主要负责确保数据格式的统一性，无需复杂的转换。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化投射器。
        :param config: 环境配置，可能包含未来扩展所需的参数。
        """
        self.config = config

    def project_observation(self, obs: Any) -> str:
        """
        将环境的观测投射为字符串。
        在JarvisEnv中，观测直接就是任务的prompt。
        
        :param obs: 来自JarvisEnv.reset()的原始观测。
        :return: 格式化后的字符串观测。
        """
        if not isinstance(obs, str):
            # 做一个类型检查以保证健壮性
            return str(obs)
        return obs

    def project_action(self, action: Any) -> str:
        """
        将智能体的动作投射为环境可以执行的字符串。
        在JarvisEnv中，动作就是需要执行的高级指令（prompt）。
        
        :param action: 来自语言模型的动作。
        :return: 格式化为字符串的动作。
        """
        if not isinstance(action, str):
            # 同样进行类型检查
            return str(action)
        return action