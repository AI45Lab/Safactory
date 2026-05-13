import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qwen_vl_utils import process_vision_info
from slime.utils.processing_utils import encode_image_for_rollout_engine


logger = logging.getLogger(__name__)
THINK_BLOCK_RE = re.compile(r"\s*<think>.*?</think>\s*", re.DOTALL)

BASE_CHAT_HISTORY = []


@dataclass
class MessageNode:
    raw_message: Optional[Dict[str, Any]]
    model_input_message: Optional[Dict[str, Any]]
    children: List["MessageNode"] = field(default_factory=list)
    delta_message_str: str = ""
    delta_tokens: List[int] = field(default_factory=list)
    delta_response_mask: List[int] = field(default_factory=list)
    delta_images: List[Any] = field(default_factory=list)
    delta_image_data: List[str] = field(default_factory=list)
    delta_mm_train_inputs: Optional[Dict[str, Any]] = None


@dataclass
class PreparedPrompt:
    node: MessageNode
    model_input_messages: List[Dict[str, Any]]
    messages_str: str
    input_ids: List[int]
    image_data: List[str]


class TrajectoryMaskBuilder:
    def __init__(self, tokenizer, processor: Any = None) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.session_roots: Dict[str, MessageNode] = {}
        if BASE_CHAT_HISTORY:
            self.base_messages_str = self.tokenizer.apply_chat_template(
                BASE_CHAT_HISTORY,
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            self.base_messages_str = ""
        self.generation_tokens = self._init_generation_tokens()
        self.suffix = self._init_suffix_tokens()

    def _init_generation_tokens(self) -> List[int]:
        test_conversation = BASE_CHAT_HISTORY + [{"role": "user", "content": "hi"}]
        without_gen = self.tokenizer.apply_chat_template(
            test_conversation,
            add_generation_prompt=False,
            tokenize=False,
        )
        with_gen = self.tokenizer.apply_chat_template(
            test_conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        generation_prompt = with_gen[len(without_gen) :]
        return list(self.tokenizer.encode(generation_prompt, add_special_tokens=False))

    def _init_suffix_tokens(self) -> List[int]:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            return []

        test_conversation = BASE_CHAT_HISTORY + [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "response"},
        ]
        test_tokens = self.tokenizer.apply_chat_template(
            test_conversation,
            add_generation_prompt=False,
            tokenize=True,
        )
        for idx in range(len(test_tokens) - 1, -1, -1):
            if test_tokens[idx] == eos_id:
                return list(test_tokens[idx + 1 :])
        return []

    def _strip_think_blocks(self, text: str) -> str:
        return THINK_BLOCK_RE.sub("", text)

    def _compare_text(self, left: str, right: str, *, ignore_think: bool) -> bool:
        left_value = left
        right_value = right
        if ignore_think:
            left_value = self._strip_think_blocks(left_value)
            right_value = self._strip_think_blocks(right_value)
        return left_value == right_value

    def _content_equals(self, left: Any, right: Any, *, ignore_think: bool) -> bool:
        if isinstance(left, str) or isinstance(right, str):
            return (
                isinstance(left, str)
                and isinstance(right, str)
                and self._compare_text(left, right, ignore_think=ignore_think)
            )

        if isinstance(left, list) or isinstance(right, list):
            if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
                return False
            return all(
                self._content_equals(left_item, right_item, ignore_think=ignore_think)
                for left_item, right_item in zip(left, right)
            )

        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                return False

            left_image = left.get("image")
            if not isinstance(left_image, str):
                left_image_url = left.get("image_url")
                if isinstance(left_image_url, dict):
                    left_image = left_image_url.get("url")
                elif isinstance(left_image_url, str):
                    left_image = left_image_url
            right_image = right.get("image")
            if not isinstance(right_image, str):
                right_image_url = right.get("image_url")
                if isinstance(right_image_url, dict):
                    right_image = right_image_url.get("url")
                elif isinstance(right_image_url, str):
                    right_image = right_image_url
            if left_image is not None or right_image is not None:
                return left_image is not None and right_image is not None and left_image == right_image

            left_text = left.get("text")
            if not isinstance(left_text, str):
                left_text = left.get("content")
            right_text = right.get("text")
            if not isinstance(right_text, str):
                right_text = right.get("content")
            if left_text is not None or right_text is not None:
                return (
                    isinstance(left_text, str)
                    and isinstance(right_text, str)
                    and self._compare_text(left_text, right_text, ignore_think=ignore_think)
                )

            return left == right

        return left == right

    def _message_matches(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if left.get("role") != right.get("role"):
            return False

        left_meta = {k: v for k, v in left.items() if k != "content"}
        right_meta = {k: v for k, v in right.items() if k != "content"}
        if left_meta != right_meta:
            return False

        return self._content_equals(
            left.get("content"),
            right.get("content"),
            ignore_think=left.get("role") == "assistant",
        )

    def _convert_content_item_for_tokenization(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        image_url_val = item.get("image_url")
        if item.get("type") == "image_url" or image_url_val is not None:
            image_value: Any = None
            if isinstance(image_url_val, dict):
                image_value = image_url_val.get("url")
            elif image_url_val is not None:
                image_value = image_url_val
            elif "image" in item:
                image_value = item.get("image")

            if isinstance(image_value, str):
                converted = dict(item)
                converted["type"] = "image"
                converted["image"] = image_value
                converted.pop("image_url", None)
                return converted

        return item

    def _message_for_model_input(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(message, dict):
            return message

        model_input_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            model_input_message["content"] = [self._convert_content_item_for_tokenization(item) for item in content]
        return model_input_message

    def _build_mm_inputs(
        self,
        text: str,
        images: List[Any],
    ) -> Tuple[List[int], Optional[Dict[str, Any]]]:
        if self.processor is None or not images:
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            return list(input_ids), None

        proc_out = self.processor(text=text, images=images, return_tensors="pt")
        raw_input_ids = proc_out["input_ids"][0]
        input_ids = raw_input_ids.tolist() if hasattr(raw_input_ids, "tolist") else list(raw_input_ids)
        mm_train_inputs = {
            "pixel_values": proc_out.get("pixel_values"),
            "image_grid_thw": proc_out.get("image_grid_thw"),
        }
        return list(input_ids), mm_train_inputs

    def _render_message_delta_str(self, model_input_message: Dict[str, Any]) -> str:
        single_message_chat_template_str = self.tokenizer.apply_chat_template(
            BASE_CHAT_HISTORY + [model_input_message],
            add_generation_prompt=False,
            tokenize=False,
        )
        if not single_message_chat_template_str.startswith(self.base_messages_str):
            raise ValueError("failed to extract single-message template fragment")
        return single_message_chat_template_str[len(self.base_messages_str) :]

    def _build_mm_train_inputs_for_images(self, images: List[Any]) -> Optional[Dict[str, Any]]:
        if self.processor is None or not images:
            return None

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            proc_out = image_processor(images=images, return_tensors="pt")
        else:
            try:
                proc_out = self.processor(images=images, return_tensors="pt")
            except TypeError:
                proc_out = self.processor(text="", images=images, return_tensors="pt")

        return {
            "pixel_values": proc_out.get("pixel_values"),
            "image_grid_thw": proc_out.get("image_grid_thw"),
        }

    def _concat_mm_value(self, left: Any, right: Any) -> Any:
        if left is None:
            return right
        if right is None:
            return left

        left_module = getattr(left.__class__, "__module__", "")
        right_module = getattr(right.__class__, "__module__", "")
        if left_module.startswith("torch") and right_module.startswith("torch"):
            import torch

            return torch.cat((left, right), dim=0)

        try:
            import numpy as np

            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                return np.concatenate((left, right), axis=0)
        except Exception:
            pass

        if isinstance(left, list) and isinstance(right, list):
            return list(left) + list(right)

        raise TypeError(f"Unsupported mm input value types: {type(left)!r}, {type(right)!r}")

    def _concat_mm_train_inputs(
        self,
        left: Optional[Dict[str, Any]],
        right: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if left is None:
            return right
        if right is None:
            return left

        return {
            "pixel_values": self._concat_mm_value(left.get("pixel_values"), right.get("pixel_values")),
            "image_grid_thw": self._concat_mm_value(left.get("image_grid_thw"), right.get("image_grid_thw")),
        }

    def _match_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[
        MessageNode,
        int,
        List[Dict[str, Any]],
        str,
        List[int],
        List[int],
        List[Any],
        List[str],
        Optional[Dict[str, Any]],
    ]:
        node = self.session_roots.get(session_id)
        if node is None:
            node = MessageNode(raw_message=None, model_input_message=None)
            self.session_roots[session_id] = node
        matched = 0
        model_input_messages: List[Dict[str, Any]] = []
        messages_str = ""
        tokens: List[int] = []
        response_mask: List[int] = []
        images: List[Any] = []
        image_data: List[str] = []
        mm_train_inputs: Optional[Dict[str, Any]] = None

        for message in messages:
            child = None
            for candidate in reversed(node.children):
                if candidate.raw_message is None:
                    continue
                if self._message_matches(candidate.raw_message, message):
                    child = candidate
                    break
            if child is None:
                break
            if child.model_input_message is not None:
                model_input_messages.append(child.model_input_message)
            messages_str += child.delta_message_str
            tokens.extend(child.delta_tokens)
            response_mask.extend(child.delta_response_mask)
            images.extend(child.delta_images)
            image_data.extend(child.delta_image_data)
            mm_train_inputs = self._concat_mm_train_inputs(mm_train_inputs, child.delta_mm_train_inputs)
            node = child
            matched += 1

        return (
            node,
            matched,
            model_input_messages,
            messages_str,
            tokens,
            response_mask,
            images,
            image_data,
            mm_train_inputs,
        )

    def _add_prompt_message(
        self,
        parent: MessageNode,
        raw_message: Dict[str, Any],
        model_input_messages: List[Dict[str, Any]],
        messages_str: str,
        tokens: List[int],
        images: List[Any],
        image_data: List[str],
    ) -> Tuple[MessageNode, List[Dict[str, Any]], str, List[int], List[Any], List[str]]:
        model_input_message = self._message_for_model_input(raw_message)
        next_model_input_messages = list(model_input_messages)
        next_model_input_messages.append(model_input_message)

        new_images, _ = process_vision_info([model_input_message])
        new_images = list(new_images or [])
        new_image_data = [encode_image_for_rollout_engine(img) for img in new_images]
        next_images = list(images)
        next_images.extend(new_images)
        next_image_data = list(image_data)
        next_image_data.extend(new_image_data)
        delta_mm_train_inputs = self._build_mm_train_inputs_for_images(new_images)

        delta_message_str = self._render_message_delta_str(model_input_message)
        next_messages_str = messages_str + delta_message_str
        delta_tokens, _ = self._build_mm_inputs(delta_message_str, new_images)
        delta_tokens = list(delta_tokens)
        combined_tokens = list(tokens)
        combined_tokens.extend(delta_tokens)
        delta_response_mask = [0] * len(delta_tokens)

        node = MessageNode(
            raw_message=raw_message,
            model_input_message=model_input_message,
            delta_message_str=delta_message_str,
            delta_tokens=delta_tokens,
            delta_response_mask=delta_response_mask,
            delta_images=new_images,
            delta_image_data=new_image_data,
            delta_mm_train_inputs=delta_mm_train_inputs,
        )
        parent.children.append(node)
        return (
            node,
            next_model_input_messages,
            next_messages_str,
            combined_tokens,
            next_images,
            next_image_data,
        )

    def _append_assistant_message(
        self,
        parent: MessageNode,
        output_ids: List[int],
        assistant_text: str,
        finish_reason: Optional[str],
    ) -> MessageNode:
        del finish_reason
        assistant_message = {"role": "assistant", "content": assistant_text}
        model_input_message = self._message_for_model_input(assistant_message)
        delta_message_str = self._render_message_delta_str(model_input_message)
        delta_tokens = list(self.generation_tokens) + list(output_ids) + list(self.suffix)
        delta_response_mask = [0] * len(self.generation_tokens) + [1] * len(output_ids) + [0] * len(self.suffix)

        node = MessageNode(
            raw_message=assistant_message,
            model_input_message=model_input_message,
            delta_message_str=delta_message_str,
            delta_tokens=delta_tokens,
            delta_response_mask=delta_response_mask,
            delta_images=[],
            delta_image_data=[],
            delta_mm_train_inputs=None,
        )
        parent.children.append(node)
        return node

    def _ensure_path(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[MessageNode, List[Dict[str, Any]], str, List[int], List[Any], List[str]]:
        node, matched, model_input_messages, messages_str, tokens, _response_mask, images, image_data, _mm_train_inputs = self._match_prefix(
            session_id,
            messages,
        )
        for message in messages[matched:]:
            node, model_input_messages, messages_str, tokens, images, image_data = self._add_prompt_message(
                node,
                message,
                model_input_messages,
                messages_str,
                tokens,
                images,
                image_data,
            )
        return node, model_input_messages, messages_str, tokens, images, image_data

    def prepare_generate_input(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> PreparedPrompt:
        node, model_input_messages, messages_str, tokens, _images, image_data = self._ensure_path(
            session_id,
            messages,
        )
        input_ids = list(tokens)
        input_ids.extend(self.generation_tokens)
        return PreparedPrompt(
            node=node,
            model_input_messages=model_input_messages,
            messages_str=messages_str,
            input_ids=input_ids,
            image_data=image_data,
        )

    def record_generation(
        self,
        prepared_prompt: PreparedPrompt,
        output_ids: List[int],
        output_logprobs: List[List[Any]],
        assistant_text: str,
        finish_reason: Optional[str] = None,
    ) -> MessageNode:
        del output_logprobs
        return self._append_assistant_message(
            parent=prepared_prompt.node,
            output_ids=list(output_ids),
            assistant_text=assistant_text,
            finish_reason=finish_reason,
        )

    def get_training_info(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ):
        node, matched, _model_input_messages, messages_str, tokens, response_mask, images, image_data, mm_train_inputs = self._match_prefix(
            session_id,
            messages,
        )
        if matched != len(messages) or node.raw_message is None:
            logger.warning(
                "get_training_info failed: session=%s, has_data=%s, matched=%s, expected=%s",
                session_id,
                session_id in self.session_roots and bool(self.session_roots[session_id].children),
                matched,
                len(messages),
            )
            return [], [], [], "", None

        if mm_train_inputs is None and self.processor is not None and images:
            mm_train_inputs = self._build_mm_train_inputs_for_images(images)

        return (
            tokens,
            response_mask,
            image_data,
            messages_str,
            mm_train_inputs,
        )

    def clear_session(self, session_id: str) -> None:
        self.session_roots.pop(session_id, None)
