import json5

from tinyAgent.LLM import ZhipuChat
from tinyAgent.tool import Tools

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Agent:
    def __init__(self, path: str = '') -> None:
        self.path = path
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.model = ZhipuChat()

    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = '', ''
        i = text.rfind('Action:')
        j = text.rfind('Action Input:')
        k = text.rfind('Observation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + 'Observation:'  # Add it back.
                k = text.rfind('Observation:')
            plugin_name = text[i + len('Action:'): j].strip()
            plugin_args = text[j + len('Action Input:'): k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name, plugin_args):
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            return 'Observation:' + self.tool.google_search_new(**plugin_args)

    def text_completion(self, text, history=[]):
        if len(history) == 0:
            history.append({'role': 'system', 'content': self.system_prompt})
        text = "Question:" + text
        response, his = self.model.chat(text, history)
        print(response)
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            search_result = self.call_plugin(plugin_name, plugin_args)
            # print(search_result)
            response += search_result
            response, his = self.model.chat(response, history)
        return response, his


if __name__ == '__main__':
    agent = Agent()
    prompt = agent.build_system_input()
    print(prompt)
