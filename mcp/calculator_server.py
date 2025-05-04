# basic import 
import hashlib
import json
import sqlite3
from markitdown import MarkItDown
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp.server import Server
from starlette.routing import Mount, Route
import tqdm
import uvicorn
from mcp import types
from PIL import Image as PILImage
import math
import sys
import time
import subprocess
from rich.console import Console
from rich.panel import Panel
import requests
from fastapi import FastAPI, Request, Body
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from pathlib import Path
import faiss

from data_model import (
    AddInput, AddListInput, AddListOutput, AddOutput, CreateThumbnailInput, OpenKeynoteOutput, PythonCodeInput, PythonCodeOutput, ShellCommandInput, SubtractInput, SubtractOutput,
    MultiplyInput, MultiplyOutput, DivideInput, DivideOutput,
    PowerInput, PowerOutput, SqrtInput, SqrtOutput,
    CbrtInput, CbrtOutput, FactorialInput, FactorialOutput,
    LogInput, LogOutput, RemainderInput, RemainderOutput,
    TrigInput, TrigOutput, MineInput, MineOutput,
    ShowReasoningInput, StringToAsciiInput, StringToAsciiOutput,
    ExponentialSumInput, ExponentialSumOutput,
    FibonacciInput, FibonacciOutput,
    KeynoteRectangleInput, KeynoteRectangleOutput,
    KeynoteTextInput, KeynoteTextOutput
)
# from win32api import GetSystemMetrics

console = Console()
# instantiate an MCP server client
mcp = FastMCP("Calculator", settings= {"host": "127.0.0.1", "port": 7172})

app = FastAPI()

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40
ROOT = Path(__file__).parent.resolve()

# Add CORS middleware to allow requests from Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you should specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define query request model
class QueryRequest(BaseModel):
    query: str

# Define page request model
class PageRequest(BaseModel):
    url: str
    title: str
    html: str

# New endpoint to handle queries from Chrome extension
# @app.post("/api/query")
# async def handle_query(request: QueryRequest):
#     query = request.query
#     print(f"Received query: {query}")
    
#     # Use the AgentService to process the query
#     try:
#         from agent_service import agent_service
        
#         # Process the query using our agent service
#         result = await agent_service.process_query(query)
#         return {"result": result}
    
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         print(f"Error processing query: {e}")
#         return {"result": f"Error processing query: {str(e)}"}

# New endpoint to handle page content from Chrome extension
# @app.post("/api/add-page")
# async def add_page(request: PageRequest):
#     print(f"Received page: {request.url}")
    
#     try:
#         # Process the HTML content
#         chunks_processed = process_html(request.url, request.title, request.html)
        
#         return {
#             "message": f"Page added successfully! Processed {chunks_processed} chunks.",
#             "status": "success"
#         }
    
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         print(f"Error processing page: {e}")
#         return {
#             "message": f"Error processing page: {str(e)}",
#             "status": "error"
#         }

# DEFINE TOOLS

#addition tool
@mcp.tool()
def add(input: AddInput) -> AddOutput:
    """Add two numbers"""
    print("CALLED: add(input: AddInput) -> AddOutput:")
    return AddOutput(result=(input.a + input.b))

@mcp.tool()
def add_list(input: AddListInput) -> AddListOutput:
    """Add all numbers in a list"""
    print("CALLED: add_list(input: AddListInput) -> AddListOutput:")
    return AddListOutput(result=sum(input.l))

# subtraction tool
@mcp.tool()
def subtract(input: SubtractInput) -> SubtractOutput:
    """Subtract two numbers"""
    print("CALLED: subtract(input: SubtractInput) -> SubtractOutput:")
    return SubtractOutput(result=(input.a - input.b))

# multiplication tool
@mcp.tool()
def multiply(input: MultiplyInput) -> MultiplyOutput:
    """Multiply two numbers"""
    print("CALLED: multiply(input: MultiplyInput) -> MultiplyOutput:")
    return MultiplyOutput(result=(input.a * input.b))

#  division tool
@mcp.tool() 
def divide(input: DivideInput) -> DivideOutput:
    """Divide two numbers"""
    print("CALLED: divide(input: DivideInput) -> DivideOutput:")
    return DivideOutput(result=(input.a / input.b))

# power tool
@mcp.tool()
def power(input: PowerInput) -> PowerOutput:
    """Power of two numbers"""
    print("CALLED: power(input: PowerInput) -> PowerOutput:")
    return PowerOutput(result=int(input.a ** input.b))

# square root tool
@mcp.tool()
def sqrt(input: SqrtInput) -> SqrtOutput:
    """Square root of a number"""
    print("CALLED: sqrt(input: SqrtInput) -> SqrtOutput:")
    return SqrtOutput(result=float(input.a ** 0.5))

# cube root tool
@mcp.tool()
def cbrt(input: CbrtInput) -> CbrtOutput:
    """Cube root of a number"""
    print("CALLED: cbrt(input: CbrtInput) -> CbrtOutput:")
    return CbrtOutput(result=float(input.a ** (1/3)))

# factorial tool
@mcp.tool()
def factorial(input: FactorialInput) -> FactorialOutput:
    """Factorial of a number"""
    print("CALLED: factorial(input: FactorialInput) -> FactorialOutput:")
    return FactorialOutput(result=int(math.factorial(input.a)))

# log tool
@mcp.tool()
def log(input: LogInput) -> LogOutput:
    """Log of a number"""
    print("CALLED: log(input: LogInput) -> LogOutput:")
    return LogOutput(result=float(math.log(input.a)))

# remainder tool
@mcp.tool()
def remainder(input: RemainderInput) -> RemainderOutput:
    """Remainder of two numbers division"""
    print("CALLED: remainder(input: RemainderInput) -> RemainderOutput:")
    return RemainderOutput(result=int(input.a % input.b))

# sin tool
@mcp.tool()
def sin(input: TrigInput) -> TrigOutput:
    """Sin of a number"""
    print("CALLED: sin(input: TrigInput) -> TrigOutput:")
    return TrigOutput(result=float(math.sin(input.a)))

# cos tool
@mcp.tool()
def cos(input: TrigInput) -> TrigOutput:
    """Cos of a number"""
    print("CALLED: cos(input: TrigInput) -> TrigOutput:")
    return TrigOutput(result=float(math.cos(input.a)))

# tan tool
@mcp.tool()
def tan(input: TrigInput) -> TrigOutput:
    """Tan of a number"""
    print("CALLED: tan(input: TrigInput) -> TrigOutput:")
    return TrigOutput(result=float(math.tan(input.a)))

# @mcp.tool()
# def calculate(expression: str) -> TextContent:
#     """Calculate the result of an expression"""
#     console.print("[blue]FUNCTION CALL:[/blue] calculate()")
#     console.print(f"[blue]Expression:[/blue] {expression}")
#     try:
#         result = eval(expression)
#         console.print(f"[green]Result:[/green] {result}")
#         return TextContent(
#             type="text",
#             text=str(result)
#         )
#     except Exception as e:
#         console.print(f"[red]Error:[/red] {str(e)}")
#         return TextContent(
#             type="text",
#             text=f"Error: {str(e)}"
#         )

# mine tool
@mcp.tool()
def mine(input: MineInput) -> MineOutput:
    """Special mining tool"""
    print("CALLED: mine(input: MineInput) -> MineOutput:")
    return MineOutput(result=int(input.a - input.b - input.b))

# reasoning tool
@mcp.tool()
def show_reasoning(input: ShowReasoningInput) -> TextContent:
    """Show the step-by-step reasoning process"""
    console.print("[blue]FUNCTION CALL:[/blue] show_reasoning()")
    for i, step in enumerate(input.steps, 1):
        console.print(Panel(
            f"{step}",
            title=f"Step {i}",
            border_style="cyan"
        ))
    
    # Create a TextContent object
    return TextContent(type="text", text="Reasoning shown")
    

@mcp.tool()
def create_thumbnail(input: CreateThumbnailInput) -> Image:
    """Create a thumbnail from an image"""
    print("CALLED: create_thumbnail(image_path: str) -> Image:")
    img = PILImage.open(input.image_path)
    img.thumbnail((100, 100))
    # return CreateThumbnailOutput(result=Image(data=img.tobytes(), format="png"))
    return Image(data=img.tobytes(), format="png")

@mcp.tool()
def strings_to_chars_to_int(input: StringToAsciiInput) -> StringToAsciiOutput:
    """Return the ASCII values of the characters in a word"""
    print("CALLED: strings_to_chars_to_int(input: StringToAsciiInput) -> StringToAsciiOutput:")
    return StringToAsciiOutput(result=[int(ord(char)) for char in input.string])

@mcp.tool()
def int_list_to_exponential_sum(input: ExponentialSumInput) -> ExponentialSumOutput:
    """Return sum of exponentials of numbers in a list"""
    print("CALLED: int_list_to_exponential_sum(input: ExponentialSumInput) -> ExponentialSumOutput:")
    return ExponentialSumOutput(result=sum(math.exp(i) for i in input.int_list))

@mcp.tool()
def fibonacci_numbers(input: FibonacciInput) -> FibonacciOutput:
    """Return the first n Fibonacci Numbers"""
    print("CALLED: fibonacci_numbers(input: FibonacciInput) -> FibonacciOutput:")
    if input.n <= 0:
        return FibonacciOutput(result=[])
    fib_sequence = [0, 1]
    for _ in range(2, input.n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return FibonacciOutput(result=fib_sequence[:input.n])

@mcp.tool()
def open_keynote() -> OpenKeynoteOutput:
    """Opens the keynote app and creates a new document in the macbook. Returns True if successful, False otherwise."""
    print("CALLED: open_keynote() -> OpenKeynoteOutput")
    apple_script = '''
    tell application "Keynote"
        activate
        set thisDocument to make new document with properties {document theme:theme "White"}
        tell thisDocument
            set base slide of the first slide to master slide "Blank"
        end tell
    end tell
    '''
    result = subprocess.run(["osascript", "-e", apple_script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return OpenKeynoteOutput(success = False)
    print("Keynote opened and new document created.")
    return OpenKeynoteOutput(success = True)

@mcp.tool()
def draw_rectangle_in_keynote(input: KeynoteRectangleInput = KeynoteRectangleInput(shapeHeight=100, shapeWidth=100)) -> KeynoteRectangleOutput:
    """Draws a rectangle in keynote app of the provided size. Returns True if rectangle is drawn successfully, False otherwise."""
    print("CALLED: draw_rectangle_in_keynote(input: KeynoteRectangleInput) -> KeynoteRectangleOutput:")
    apple_script = f'''
    tell application "Keynote"
        tell document 1
            set docWidth to its width
            set docHeight to its height
            set x to (docWidth - {{{input.shapeWidth}}}) div 2
            set y to (docHeight - {{{input.shapeHeight}}}) div 2
            tell slide 1
                set newRectangle to make new shape with properties {{position:{{x, y}}, width:{input.shapeWidth}, height:{input.shapeHeight}}}
            end tell
        end tell
    end tell
    '''
    result = subprocess.run(["osascript", "-e", apple_script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Draw rectangle error:", result.stderr)
        return KeynoteRectangleOutput(success=False)
    print("Rectangle drawn on the slide.")
    return KeynoteRectangleOutput(success=True)

@mcp.tool()
def add_text_to_keynote_shape(input: KeynoteTextInput) -> KeynoteTextOutput:
    """Adds a text to the shape drawn in keynote. Return True if text was added successfully, False otherwise."""
    print("CALLED: add_text_to_keynote_shape(input: KeynoteTextInput) -> KeynoteTextOutput:")
    apple_script = f'''
    tell application "Keynote"
        tell document 1
            tell slide 1
                set the object text of the shape 1 to "{input.text}"
            end tell
        end tell
    end tell
    '''
    result = subprocess.run(["osascript", "-e", apple_script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Add text error:", result.stderr)
        return KeynoteTextOutput(success=False)
    print("Text added to the rectangle.")
    return KeynoteTextOutput(success=True)



# New Tools
from io import StringIO
import sys
import math
import subprocess

@mcp.tool()
def run_python_sandbox(input: PythonCodeInput) -> PythonCodeOutput:
    """Run math code in Python sandbox. Usage: run_python_sandbox|input={"code": "result = math.sqrt(49)"}"""
    import sys, io
    import math

    allowed_globals = {
        "__builtins__": __builtins__  # Allow imports like in executor.py
    }

    local_vars = {}

    # Capture print output
    stdout_backup = sys.stdout
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    try:
        exec(input.code, allowed_globals, local_vars)
        sys.stdout = stdout_backup
        result = local_vars.get("result", output_buffer.getvalue().strip() or "Executed.")
        return PythonCodeOutput(result=str(result))
    except Exception as e:
        sys.stdout = stdout_backup
        return PythonCodeOutput(result=f"ERROR: {e}")


@mcp.tool()
def run_shell_command(input: ShellCommandInput) -> PythonCodeOutput:
    """Run a safe shell command. Usage: run_shell_command|input={"command": "ls"}"""
    allowed_commands = ["ls", "cat", "pwd", "df", "whoami"]

    tokens = input.command.strip().split()
    if tokens[0] not in allowed_commands:
        return PythonCodeOutput(result="Command not allowed.")

    try:
        result = subprocess.run(
            input.command, shell=True,
            capture_output=True, timeout=3
        )
        output = result.stdout.decode() or result.stderr.decode()
        return PythonCodeOutput(result=output.strip())
    except Exception as e:
        return PythonCodeOutput(result=f"ERROR: {e}")


@mcp.tool()
def run_sql_query(input: PythonCodeInput) -> PythonCodeOutput:
    """Run safe SELECT-only SQL query. Usage: run_sql_query|input={"code": "SELECT * FROM users LIMIT 5"}"""
    if not input.code.strip().lower().startswith("select"):
        return PythonCodeOutput(result="Only SELECT queries allowed.")

    try:
        conn = sqlite3.connect("example.db")
        cursor = conn.cursor()
        cursor.execute(input.code)
        rows = cursor.fetchall()
        result = "\n".join(str(row) for row in rows)
        return PythonCodeOutput(result=result or "No results.")
    except Exception as e:
        return PythonCodeOutput(result=f"ERROR: {e}")

# DEFINE RESOURCES

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    print("CALLED: get_greeting(name: str) -> str:")
    return f"Hello, {name}!"


# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    print("CALLED: review_code(code: str) -> str:")
    return f"Please review this code:\n\n{code}"
    

@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

def start_sse():
    mcp_server = mcp._mcp_server  # noqa: WPS437
    import argparse
    from pdb import set_trace

    # set_trace()
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=7172, help='Port to listen on')
    args = parser.parse_args()
    print("SSE args set.")

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)
    app.mount("/", starlette_app)

    uvicorn.run(app, host=args.host, port=args.port) 

@app.get("/")
def read_root():
    return {"Hello": "Worlddd"}

@app.get("/mcp")
async def get_capabilites():
    return await mcp.list_tools()

def get_embedding(text: str) -> np.ndarray:
    response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level}: {message}\n")
    sys.stderr.flush()

def process_documents():
    """Process documents and create FAISS index"""
    mcp_log("INFO", "Indexing documents with MarkItDown...")
    ROOT = Path(__file__).parent.resolve()
    DOC_PATH = ROOT / "documents"
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    def file_hash(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None
    all_embeddings = []
    converter = MarkItDown()

    for file in DOC_PATH.glob("*.*"):
        fhash = file_hash(file)
        if file.name in CACHE_META and CACHE_META[file.name] == fhash:
            mcp_log("SKIP", f"Skipping unchanged file: {file.name}")
            continue

        mcp_log("PROC", f"Processing: {file.name}")
        try:
            result = converter.convert(str(file))
            markdown = result.text_content
            chunks = list(chunk_text(markdown))
            embeddings_for_file = []
            new_metadata = []
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {file.name}")):
                embedding = get_embedding(chunk)
                embeddings_for_file.append(embedding)
                new_metadata.append({"doc": file.name, "chunk": chunk, "chunk_id": f"{file.stem}_{i}"})
            if embeddings_for_file:
                if index is None:
                    dim = len(embeddings_for_file[0])
                    index = faiss.IndexFlatL2(dim)
                index.add(np.stack(embeddings_for_file))
                metadata.extend(new_metadata)
            CACHE_META[file.name] = fhash
        except Exception as e:
            mcp_log("ERROR", f"Failed to process {file.name}: {e}")

    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_FILE))
        mcp_log("SUCCESS", "Saved FAISS index and metadata")
    else:
        mcp_log("WARN", "No new documents or updates to process.")

def process_html(url, title, html_content):
    """Process HTML content from a webpage and add to FAISS index"""
    mcp_log("INFO", f"Processing HTML from: {url}")
    
    # Setup paths and files
    ROOT = Path(__file__).parent.resolve()
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"
    
    # Create a URL-safe filename
    safe_url = url.replace("://", "_").replace("/", "_").replace(".", "_")
    temp_file_path = INDEX_CACHE / f"temp_{safe_url}.html"
    
    # Write HTML to temporary file
    # with open(temp_file_path, "w", encoding="utf-8") as f:
    #     f.write(html_content)
    
    # Calculate hash of content
    content_hash = hashlib.md5(html_content.encode("utf-8")).hexdigest()
    
    # Load existing metadata and cache
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    
    # Check if we've already processed this URL with the same content
    if url in CACHE_META and CACHE_META[url] == content_hash:
        mcp_log("SKIP", f"Skipping unchanged URL: {url}")
        # Clean up temp file
        # temp_file_path.unlink(missing_ok=True)
        return 0
    
    try:
        # Process the HTML
        converter = MarkItDown()
        # result = converter.convert(str(temp_file_path))
        result = converter.convert_url(url)
        markdown = result.text_content
        
        # Create chunks of text
        chunks = list(chunk_text(markdown))
        
        # Load or create FAISS index
        index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() and Path(INDEX_FILE).is_file() else None
        
        # Process each chunk and create embeddings
        embeddings = []
        new_metadata = []
        
        for i, chunk in enumerate(tqdm.tqdm(chunks, desc=f"Embedding {url}")):
            embedding = get_embedding(chunk)
            embeddings.append(embedding)
            new_metadata.append({
                "url": url,
                "title": title,
                "chunk": chunk,
                "chunk_id": f"{safe_url}_{i}"
            })
        
        # If we have embeddings, add them to the index
        if embeddings:
            embeddings_array = np.stack(embeddings)
            
            if index is None:
                # Create a new index if none exists
                dim = len(embeddings[0])
                index = faiss.IndexFlatL2(dim)
            
            # Add embeddings to index
            index.add(embeddings_array)
            
            # Add metadata
            metadata.extend(new_metadata)
            
            # Update cache
            CACHE_META[url] = content_hash
            
            # Save everything
            METADATA_FILE.write_text(json.dumps(metadata, indent=2))
            CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
            faiss.write_index(index, str(INDEX_FILE))
            
            mcp_log("SUCCESS", f"Indexed {len(chunks)} chunks from {url}")
        
        # Clean up temp file
        # temp_file_path.unlink(missing_ok=True)
        
        return len(chunks)
        
    except Exception as e:
        mcp_log("ERROR", f"Failed to process {url}: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp file
        temp_file_path.unlink(missing_ok=True)
        
        raise e

def search_index(query, k=5):
    """Search the FAISS index for similar content to the query"""
    ROOT = Path(__file__).parent.resolve()
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        return {"results": [], "error": "No index available. Try adding some pages first."}
    
    try:
        # Load index and metadata
        index = faiss.read_index(str(INDEX_FILE))
        metadata = json.loads(METADATA_FILE.read_text())
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Reshape for search
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = index.search(query_embedding, k=k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(metadata):  # Valid index
                item = metadata[idx]
                result = {
                    "score": float(distances[0][i]),
                    "metadata": item
                }
                results.append(result)
        
        return {"results": results}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"results": [], "error": str(e)}

# Add endpoint for searching the index
# @app.post("/api/search")
# async def search_content(request: QueryRequest):
#     query = request.query
#     print(f"Searching for: {query}")
    
#     try:
#         results = search_index(query)
#         return results
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return {"results": [], "error": str(e)}

if __name__ == "__main__":
    # Check if running with mcp dev command
    print("STARTING")
    if len(sys.argv) > 1:
        if sys.argv[1] == "dev":
            print("STARTING without transport for dev server")
            mcp.run() 
        elif sys.argv[1] == "sse":
            sys.argv.remove("sse")
            print("STARTING sse server")
            start_sse()
     # Run without transport for dev server
    else:
        print("STARTING with stdio for direct execution")
        mcp.run(transport="stdio")
else: 
    print("starting sse...")
    start_sse()
        

 # Run with stdio for direct execution
