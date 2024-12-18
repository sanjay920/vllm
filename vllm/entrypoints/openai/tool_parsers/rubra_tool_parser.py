from typing import Dict, List, Sequence, Union
import json
import re
import logging
from uuid import uuid4

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
    ToolCall,
    FunctionCall,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.tool_parsers.tool_parser import (
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)


@ToolParserManager.register_module("rubra")
class RubraToolParser(ToolParser):
    def __init__(self):
        self.tool_pattern = r"starttoolcall(.*?)endtoolcall"
        self.tool_regex = re.compile(self.tool_pattern)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if "starttoolcall" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = self.tool_regex.finditer(model_output)
            tool_calls = []

            for match in matches:
                tool_call_json = match.group(1)
                tool_call_dict = json.loads(tool_call_json)

                tool_calls.append(
                    ToolCall(
                        type="function",
                        id=f"chatcmpl-tool-{uuid4().hex[:8]}",
                        function=FunctionCall(
                            name=tool_call_dict["name"],
                            arguments=json.dumps(tool_call_dict["arguments"]),
                        ),
                    )
                )

            content = model_output[: model_output.find("starttoolcall")]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content.strip() else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Rubra response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if "starttoolcall" not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            # Extract the tool call between starttoolcall and endtoolcall
            matches = list(self.tool_regex.finditer(current_text))
            if not matches:
                return None

            last_match = matches[-1]
            tool_call_json = last_match.group(1)

            try:
                tool_call_dict = json.loads(tool_call_json)
            except json.JSONDecodeError:
                return None

            # If we have a complete tool call, send it
            if "name" in tool_call_dict and "arguments" in tool_call_dict:
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=len(matches) - 1,
                            type="function",
                            id=f"chatcmpl-tool-{uuid4().hex[:8]}",
                            function=DeltaFunctionCall(
                                name=tool_call_dict["name"],
                                arguments=json.dumps(tool_call_dict["arguments"]),
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )

            return None

        except Exception:
            logger.exception("Error handling streaming tool call in Rubra")
            return None
