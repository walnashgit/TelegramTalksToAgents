from pydantic import BaseModel
from typing import List, Optional
from mcp.types import TextContent 
from mcp.server.fastmcp import Image # Ensure this import is present


class FuntionInfo(BaseModel):
    func_name: str
    param: BaseModel

# Models for addition
class AddInput(BaseModel):
    a: int
    b: int

class AddOutput(BaseModel):
    result: int

# Models for add_list
class AddListInput(BaseModel):
    l: List[int]

class AddListOutput(BaseModel):
    result: int

# Models for subtraction
class SubtractInput(BaseModel):
    a: int
    b: int

class SubtractOutput(BaseModel):
    result: int

# Models for multiplication
class MultiplyInput(BaseModel):
    a: int
    b: int

class MultiplyOutput(BaseModel):
    result: int

# Models for division
class DivideInput(BaseModel):
    a: int
    b: int

class DivideOutput(BaseModel):
    result: float

# Models for power
class PowerInput(BaseModel):
    a: int
    b: int

class PowerOutput(BaseModel):
    result: int

# Models for square root
class SqrtInput(BaseModel):
    a: int

class SqrtOutput(BaseModel):
    result: float

# Models for cube root
class CbrtInput(BaseModel):
    a: int

class CbrtOutput(BaseModel):
    result: float

# Models for factorial
class FactorialInput(BaseModel):
    a: int

class FactorialOutput(BaseModel):
    result: int

# Models for logarithm
class LogInput(BaseModel):
    a: int

class LogOutput(BaseModel):
    result: float

# Models for remainder
class RemainderInput(BaseModel):
    a: int
    b: int

class RemainderOutput(BaseModel):
    result: int

# Models for trigonometric functions
class TrigInput(BaseModel):
    a: int

class TrigOutput(BaseModel):
    result: float

# Models for mine function
class MineInput(BaseModel):
    a: int
    b: int

class MineOutput(BaseModel):
    result: int

# Models for show_reasoning
class ShowReasoningInput(BaseModel):
    steps: List[str]

class CreateThumbnailInput(BaseModel):
    image_path: str

# Models for strings_to_chars_to_int
class StringToAsciiInput(BaseModel):
    string: str

class StringToAsciiOutput(BaseModel):
    result: List[int]

# Models for int_list_to_exponential_sum
class ExponentialSumInput(BaseModel):
    int_list: List[int]

class ExponentialSumOutput(BaseModel):
    result: float

# Models for fibonacci_numbers
class FibonacciInput(BaseModel):
    n: int

class FibonacciOutput(BaseModel):
    result: List[int]

class OpenKeynoteOutput(BaseModel):
    success: bool

# Models for Keynote operations
class KeynoteRectangleInput(BaseModel):
    shapeWidth: int
    shapeHeight: int

class KeynoteRectangleOutput(BaseModel):
    success: bool

class KeynoteTextInput(BaseModel):
    text: str

class KeynoteTextOutput(BaseModel):
    success: bool

class PythonCodeInput(BaseModel):
    code: str

class PythonCodeOutput(BaseModel):
    result: str

class ShellCommandInput(BaseModel):
    command: str