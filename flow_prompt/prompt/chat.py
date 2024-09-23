from abc import abstractmethod
import base64
from dataclasses import dataclass
import logging
import typing as t
import uuid
import httpx

from flow_prompt.exceptions import ValueIsNotResolvedError
from flow_prompt.utils import resolve

logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class PromptContent:
    _type: str

    @property
    def content_type(self) -> str:
        return self._type

    @abstractmethod
    def dump(self) -> t.Dict: 
        pass

@dataclass(kw_only=True)
class TextPromptContent(PromptContent):
    _type: str = "text"
    content: str = ""

    def dump(self) -> t.Dict:
        return {
            "text": self.content,
        }


@dataclass(kw_only=True)
class ImagePromptContent(PromptContent):
    _type: str = "image"
    _mime_type: str
    _image_base64: str 
    _image_content: bytes 

    def dump(self) -> t.Dict:
        return {
            "image_base64": self._image_base64,
            "image_content": self._image_content,
            "mime_type": self._mime_type,
            "encoding": "base64",
        } 

    @classmethod
    def load_from_base64_string(cls, base64_content: str, mime_type: str):
        try: 
            content = base64.b64decode(base64_content)
            return cls(
                _image_content=content,
                _image_base64=base64_content,
                _mime_type=mime_type,
            )
        except Exception as e:
            logger.error(f"Failed to load image from base64 string {base64_content}")
            raise e

    @classmethod
    def load_from_path(cls, path: str, mime_type: str):

        try:
            with open(path, "rb") as f:
                content = f.read()
                return cls(
                    _image_content=content,
                    _image_base64=base64.b64encode(content).decode("utf-8"),
                    _mime_type=mime_type,
                )
        except Exception as e:
            logger.error(f"Failed to load image from path {path}")
            raise e

    @classmethod
    def load_from_url(cls, url: str, mime_type: str):
        try: 
            content = httpx.get(url).content
            return cls(
                _image_content=content,
                _image_base64=base64.b64encode(content).decode("utf-8"),
                _mime_type=mime_type,
            )
        except Exception as e:
            logger.error(f"Failed to load image from url {url}")
            raise e


@dataclass(kw_only=True)
class FilePromptContent(PromptContent):
    _type: str = "file"
    _mime_type: str 
    _file_content: bytes

    def dump(self) -> t.Dict:
        return {
            "mime_type": self._mime_type,
        } 

    @classmethod
    def load_from_path(cls, path: str, mime_type: str):
        try: 
            with open(path, "rb") as f:
                content=f.read()
                return cls(
                    _mime_type=mime_type,
                    _file_content=content,
                )
        except Exception as e:
            logger.error(f"Failed to load file from path {path}")
            raise e

    @classmethod
    def load_from_url(cls, url: str, mime_type: str):
        try: 
            content = httpx.get(url).content
            return cls(
                _mime_type=mime_type,
                _file_content=content,
            )
        except Exception as e:
            logger.error(f"Failed to load file from url {url}")
            raise e


class ChatMessage:
    role: str
    content_type: str
    content: t.Any
    name: t.Optional[str] = None
    tool_calls: t.Dict[str, str] = {}
    ref_name: t.Optional[str] = None
    ref_value: t.Optional[str] = None

    def __init__(self, **kwargs):
        self.role = kwargs.get("role", "user")
        self.content_type = kwargs.get("content_type", "text")
        self.content = kwargs["content"]
        self.name = kwargs.get("name")
        self.tool_calls = kwargs.get("tool_calls") or {}
        self.ref_name = kwargs.get("ref_name")
        self.ref_value = kwargs.get("ref_value")

    def is_not_empty(self):
        return bool(self.content or self.tool_calls)

    def is_empty(self):
        return not self.is_not_empty()

    def not_tool_calls(self):
        return not (self.tool_calls)

    def to_dict(self):
        result = {
            "role": self.role,
            "type": self.content_type,
            "content": self.content,
        }

        if self.name:
            result["name"] = self.name

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        if self.ref_name:
            result["ref_name"] = self.ref_name

        if self.ref_name:
            result["ref_value"] = self.ref_value

        return result

@dataclass(kw_only=True)
class ChatsEntity:
    content_type: str = "text"
    content: t.Any = None
    role: str = "user"
    name: t.Optional[str] = None
    tool_calls: t.Optional[t.Dict[str, str]] = None
    priority: int = 0
    required: bool = False
    is_multiple: bool = False
    while_fits: bool = False
    add_in_reverse_order: bool = False
    in_one_message: bool = False
    continue_if_doesnt_fit: bool = False

    add_if_fitted_labels: t.List[str] = None
    label: t.Optional[str] = None
    presentation: t.Optional[str] = None
    last_words: t.Optional[str] = None

    ref_name: t.Optional[str] = None
    ref_value: t.Optional[str] = None
    ref_image: t.Optional[str] = None
    ref_file: t.Optional[str] = None

    def __post_init__(self):
        self._uuid = uuid.uuid4().hex

    def resolve(self, context: t.Dict[str, t.Any], images: t.Optional[t.Dict[str, t.Any]] = None, 
                files: t.Optional[t.Dict[str, t.Any]] = None) -> t.List[ChatMessage]:
        if self.content_type == "image":
            if self.ref_image is None:
                raise ValueIsNotResolvedError(
                    f"Could not resolve image. Image reference is not found"
                )

            if images is None or not images[self.ref_image]:
                raise ValueIsNotResolvedError(
                    f"Could not find image reference. {self.ref_image} should exist"
                )

            content = images[self.ref_image]
            return [
             ChatMessage(
                name=self.name,
                role=self.role,
                content_type="image",
                content=content.dump(),
                tool_calls=self.tool_calls,
                ref_name=self.ref_name,
                ref_value=self.ref_value,
                )
            ]

        if self.content_type == "file":
            if self.ref_file is None:
                raise ValueIsNotResolvedError(
                    f"Could resolve file. File reference not found"
                )

            if files is None or not files[self.ref_file]:
                raise ValueIsNotResolvedError(
                    f"Could not find file reference. {self.ref_file} should exist"
                )

            content = files[self.ref_file]
            return [
             ChatMessage(
                name=self.name,
                role=self.role,
                content_type="file",
                content=content.dump(),
                tool_calls=self.tool_calls,
                ref_name=self.ref_name,
                ref_value=self.ref_value,
                )
            ]

        result = []
        content = self.content

        if self.is_multiple:
            # should be just one value like {messages} in prompt
            prompt_value = content.strip().replace("{", "").replace("}", "").strip()
            values = context.get(prompt_value, [])
            if not values:
                return []

            if not isinstance(values, list):
                raise ValueIsNotResolvedError(
                    f"Invalid value {values} for prompt {content}. Should be multiple"
                )
            else:
                # verify that values are json list of ChatMessage
                try:
                    result = [
                        ChatMessage(**({"content": c} if isinstance(c, str) else c))
                        for c in values
                    ]
                except TypeError as e:
                    raise ValueIsNotResolvedError(
                        f"Invalid value {values} for prompt {content}. Error: {e}"
                    )

            return result


        content = resolve(content, context)
        if not content:
            return []

        return [
             ChatMessage(
                name=self.name,
                role=self.role,
                content_type="text",
                content=content,
                tool_calls=self.tool_calls,
                ref_name=self.ref_name,
                ref_value=self.ref_value,
            )
        ]

    def get_values(self, context: t.Dict[str, t.Any], images: t.Optional[t.Dict[str, t.Any]] = None, 
                   files: t.Optional[t.Dict[str, t.Any]] = None) -> t.List[ChatMessage]:
        try:
            values = self.resolve(context, images, files)
        except Exception as e:
            logger.error(
                f"Error resolving prompt {self.content}, error: {e}", exc_info=True
            )
            return []
        return values

    def dump(self):
        data = {
            "content_type": self.content_type,
            "content": self.content,
            "role": self.role,
            "name": self.name,
            "tool_calls": self.tool_calls,
            "priority": self.priority,
            "required": self.required,
            "is_multiple": self.is_multiple,
            "while_fits": self.while_fits,
            "add_in_reverse_order": self.add_in_reverse_order,
            "in_one_message": self.in_one_message,
            "continue_if_doesnt_fit": self.continue_if_doesnt_fit,
            "add_if_fitted_labels": self.add_if_fitted_labels,
            "label": self.label,
            "presentation": self.presentation,
            "last_words": self.last_words,
            "ref_name": self.ref_name,
            "ref_value": self.ref_value,
            "ref_image": self.ref_image,
            "ref_file": self.ref_file,
        }
        for k, v in list(data.items()):
            if v is None:
                del data[k]
        return data

    @classmethod
    def load(cls, data):
        return cls(
            content_type=data.get("content_type"),
            content=data.get("content"),
            role=data.get("role"),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            priority=data.get("priority"),
            required=data.get("required"),
            is_multiple=data.get("is_multiple"),
            while_fits=data.get("while_fits"),
            add_in_reverse_order=data.get("add_in_reverse_order"),
            in_one_message=data.get("in_one_message"),
            continue_if_doesnt_fit=data.get("continue_if_doesnt_fit"),
            add_if_fitted_labels=data.get("add_if_fitted_labels"),
            label=data.get("label"),
            presentation=data.get("presentation"),
            last_words=data.get("last_words"),
            ref_name=data.get("ref_name"),
            ref_value=data.get("ref_value"),
            ref_image=data.get("ref_image"),
            ref_file=data.get("ref_file"),
        )
