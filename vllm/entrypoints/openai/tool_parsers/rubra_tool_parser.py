import json
import re
import logging
from typing import Dict, List, Sequence, Union
from uuid import uuid4

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
    ToolCall,
    FunctionCall,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("rubra")
class RubraToolParser(ToolParser):
    """Tool parser for Rubra format that uses starttoolcall/endtoolcall markers."""

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: List[str] = []

        self.tool_pattern = r"starttoolcall(.*?)endtoolcall"
        self.tool_regex = re.compile(self.tool_pattern)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output."""
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
        """Extract tool calls from streaming output."""
        if "starttoolcall" not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            # Extract the tool call between starttoolcall and endtoolcall
            matches = list(self.tool_regex.finditer(current_text))
            if not matches:
                return None

            last_match = matches[-1]
            tool_call_json = last_match.group(1)

            # Parse with partial JSON parser to handle incomplete JSON
            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
            try:
                tool_call_dict = partial_json_parser.loads(tool_call_json, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                return None

            # Handle new tool call
            if len(matches) > len(self.prev_tool_call_arr):
                self.current_tool_id = len(matches) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")

            # If name not sent yet, send it if available
            if not self.current_tool_name_sent:
                function_name = tool_call_dict.get("name")
                if function_name:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"chatcmpl-tool-{uuid4().hex[:8]}",
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                    return delta
                return None

            # Handle streaming arguments
            cur_arguments = tool_call_dict.get("arguments")
            prev_arguments = (
                self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if self.prev_tool_call_arr
                else None
            )

            if not cur_arguments:
                return None

            if not prev_arguments:
                # First time getting arguments
                cur_args_json = json.dumps(cur_arguments)
                if delta_text not in cur_args_json:
                    return None

                args_delta = cur_args_json[
                    : cur_args_json.index(delta_text) + len(delta_text)
                ]
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=args_delta).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += args_delta

            else:
                # Streaming additional arguments
                cur_args_json = json.dumps(cur_arguments)
                prev_args_json = json.dumps(prev_arguments)

                if cur_args_json == prev_args_json:
                    return None

                argument_diff = extract_intermediate_diff(cur_args_json, prev_args_json)
                if not argument_diff:
                    return None

                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=argument_diff
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            # Update state
            if len(matches) > len(self.prev_tool_call_arr):
                self.prev_tool_call_arr = [tool_call_dict]
            else:
                self.prev_tool_call_arr[self.current_tool_id] = tool_call_dict

            return delta

        except Exception:
            logger.exception("Error handling streaming tool call in Rubra")
            return None
