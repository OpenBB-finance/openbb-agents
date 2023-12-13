"""Load OpenBB functions at OpenAI tools for function calling in Langchain"""
import inspect
from functools import wraps
from types import ModuleType
from typing import Callable, List, Union

import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.tools import StructuredTool
from langchain.tools.base import ToolException
from langchain.vectorstores import FAISS, VectorStore
from openbb import obb
from pydantic.v1 import ValidationError, create_model
from pydantic.v1.fields import FieldInfo
from pydantic_core import PydanticUndefinedType


def create_tool_index(tools: list[StructuredTool]) -> VectorStore:
    """Create a tool index of LangChain StructuredTools."""
    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(tools)
    ]

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store


def _fetch_obb_module(openbb_command_root: str) -> ModuleType:
    module_path_split = openbb_command_root.split("/")[1:]
    module_path = ".".join(module_path_split)

    # Iteratively get module
    module = obb
    for attr in module_path.split("."):
        module = getattr(module, attr)

    return module


def _fetch_schemas(openbb_command_root: str) -> dict:
    # Ugly hack to make it compatiable with the look-up (even though we convert
    # it back) so that we have a nicer API for the user.
    module_root_path = openbb_command_root.replace("/", ".")
    schemas = {
        k.replace(".", "/"): v
        for k, v in obb.coverage.command_model.items()
        if module_root_path in k
    }
    return schemas


def _fetch_callables(openbb_command_root):
    module = _fetch_obb_module(openbb_command_root)

    if inspect.ismethod(
        module
    ):  # Handle case where a final command endpoint is passed.
        members_dict = {module.__name__: module}
    else:  # If a command root is passed instead
        members = inspect.getmembers(module)
        members_dict = {
            x[0]: x[1] for x in members if "__" not in x[0] and "_run" not in x[0]
        }

    schemas = _fetch_schemas(openbb_command_root)
    # Create callables dict, with the same key as used in the schemas
    callables = {}
    for k in schemas.keys():
        try:
            callables[k] = members_dict[k.split("/")[-1]]
        except (
            KeyError
        ):  # Sometimes we don't have a specific callable for an endpoint, so we skip.
            pass
    return callables


def _fetch_outputs(schema):
    outputs = []
    output_fields = schema["openbb"]["Data"]["fields"]
    for name, t in output_fields.items():
        if isinstance(t.annotation, type):
            type_str = t.annotation.__name__
        else:
            type_str = str(t.annotation).replace("typing.", "")
        outputs.append((name, type_str))
    return outputs


def from_schema_to_pydantic_model(model_name, schema):
    create_model_kwargs = {}
    for field, field_info in schema.items():
        field_type = field_info.annotation

        # Handle default values
        if not isinstance(field_info.default, PydanticUndefinedType):
            field_default_value = field_info.default
            new_field_info = (
                FieldInfo(  # Weird hack, because of how the default field value works
                    description=field_info.description,
                    default=field_default_value,
                )
            )
        else:
            new_field_info = FieldInfo(
                description=field_info.description,
            )
        create_model_kwargs[field] = (field_type, new_field_info)
    return create_model(model_name, **create_model_kwargs)


def return_results(func):
    """Return the results rather than the OBBject."""

    def wrapper_func(*args, **kwargs):
        try:
            result = func(*args, **kwargs).results
            encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")
            num_tokens = len(encoding.encode(str(result)))
            if num_tokens > 90000:
                raise ToolException(
                    "The returned output is too large to fit into context. Consider using another tool, or trying again with different input arguments."  # noqa: E501
                )
            return result
        # Necessary to catch general exception in this case, since we want the
        # LLM to be able to correct a bad call, if possible.
        except Exception as err:
            raise ToolException(err) from err

    return wrapper_func


def from_openbb_to_langchain_func(
    openbb_command_root: str, openbb_callable: Callable, openbb_schema: dict
) -> StructuredTool:
    func_schema = openbb_schema["openbb"]["QueryParams"]["fields"]
    # Lookup the default provider's input arguments...
    default_provider = obb.coverage.commands[openbb_command_root.replace("/", ".")][0]
    # ... and add them to the func schema.
    func_schema.update(openbb_schema[default_provider]["QueryParams"]["fields"])
    pydantic_model = from_schema_to_pydantic_model(
        model_name=f"{openbb_command_root}InputModel", schema=func_schema
    )

    outputs = _fetch_outputs(openbb_schema)
    description = openbb_callable.__doc__.split("\n")[0]
    description += "\nThe following data is available in the output:\n\n"
    description += ", ".join(e[0].replace("_", " ") for e in outputs)

    tool = StructuredTool(
        name=openbb_command_root,  # We use the command root for the name of the tool
        func=return_results(openbb_callable),
        description=description,
        args_schema=pydantic_model,
        handle_tool_error=True,
    )

    # We have to do some magic here to prevent a bad input argument from
    # breaking the langchain flow
    # https://github.com/langchain-ai/langchain/issues/13662#issuecomment-1831242057
    def handle_validation_error(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as err:
                return str(err)

        return wrapper

    # Monkey-patch the run method
    object.__setattr__(tool, "run", handle_validation_error(tool.run))

    return tool


def map_openbb_functions_to_langchain_tools(
    openbb_command_root, schemas_dict, callables_dict
):
    tools = []
    for route in callables_dict.keys():
        tool = from_openbb_to_langchain_func(
            openbb_command_root=route,
            openbb_callable=callables_dict[route],
            openbb_schema=schemas_dict[route],
        )
        tools.append(tool)
    return tools


def map_openbb_routes_to_langchain_tools(
    openbb_commands_root: Union[str, List[str]],
) -> list[StructuredTool]:
    """Map a collection of OpenBB callables from a command root to StructuredTools.

    Examples
    --------
    >>> fundamental_tools = map_openbb_collection_to_langchain_tools(
    ...     "/equity/fundamental"
    ... )
    >>> crypto_price_tools = map_openbb_collection_to_langchain_tools(
    ...     "/crypto/price"
    ... )


    """
    openbb_commands_root_list = (
        [openbb_commands_root]
        if isinstance(openbb_commands_root, str)
        else openbb_commands_root
    )

    tools: List = []
    for obb_cmd_root in openbb_commands_root_list:
        schemas = _fetch_schemas(obb_cmd_root)
        callables = _fetch_callables(obb_cmd_root)
        tools += map_openbb_functions_to_langchain_tools(
            openbb_command_root=obb_cmd_root,
            schemas_dict=schemas,
            callables_dict=callables,
        )
    return tools


def get_all_openbb_tools():
    tool_routes = list(obb.coverage.commands.keys())
    tool_routes = [
        route.replace(".", "/") for route in tool_routes if "metrics" not in route
    ]

    tools = []
    for route in tool_routes:
        schema = _fetch_schemas(route)
        callables = _fetch_callables(route)
        tools += map_openbb_functions_to_langchain_tools(route, schema, callables)
    return tools
