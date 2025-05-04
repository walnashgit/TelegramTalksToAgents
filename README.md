# S8MCP Goes to Internet

A powerful Telegram bot that leverages MCP (Model Context Protocol) servers to process and respond to user queries. The bot can connect to various MCP servers, utilize multiple tools, and provide intelligent responses to user queries.

## Features

- 🤖 Telegram Bot Integration
- 🔄 Asynchronous Query Processing
- 🌐 Multi-MCP Server Support
- 🛠️ Dynamic Tool Integration
- ⚡ Real-time Response Generation
- 🔍 Advanced Query Processing
- 📊 Rich Response Formatting

## Architecture

The project follows a modular architecture with the following key components:

1. **Telegram Bot Interface** (`telegram_bot.py`)
   - Handles user interactions
   - Manages message processing
   - Provides real-time feedback
   - Implements typing indicators

2. **Agent System** (`agent.py`)
   - Core query processing logic
   - MCP server management
   - Response generation and formatting
   - Error handling and logging

3. **Core Components** (`core/`)
   - `loop.py`: Implements the main agent processing loop
     - Manages the step-by-step execution of queries
     - Handles perception, memory retrieval, planning, and tool execution
     - Implements error handling and response formatting
   
   - `session.py`: Manages MCP server connections
     - `MCP`: Lightweight wrapper for one-time MCP tool calls
     - `MultiMCP`: Handles multiple MCP server connections
     - Supports both stdio and SSE transport methods
     - Manages tool discovery and execution across servers
   
   - `context.py`: Maintains agent state and configuration
     - Stores session information
     - Manages agent profiles and settings
     - Handles context persistence
   
   - `strategy.py`: Implements decision-making logic
     - Determines next actions based on context
     - Manages tool selection and execution order

4. **Module Components** (`modules/`)
   - `perception.py`: Extracts structured information from user input
     - Uses LLMs to identify user intent
     - Extracts entities and keywords
     - Suggests relevant tools
   
   - `memory.py`: Manages agent memory and context
     - Stores and retrieves past interactions
     - Implements memory filtering and ranking
     - Maintains session history
   
   - `action.py`: Handles tool execution and response processing
     - Parses function calls
     - Manages tool input/output
     - Processes tool responses
   
   - `decision.py`: Implements decision-making logic
     - Evaluates available options
     - Selects appropriate tools
     - Manages execution flow
   
   - `tools.py`: Manages available tools and their configurations
     - Registers and maintains tool definitions
     - Handles tool discovery and validation
   
   - `model_manager.py`: Manages LLM interactions
     - Handles model initialization
     - Manages text generation
     - Implements model-specific configurations

## Prerequisites

- Python 3.11 or higher
- Telegram Bot Token
- MCP Server configurations
- Required Python packages (listed in `pyproject.toml`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd S8MCPGoesToInternet
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   Create a `.env` file with:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

5. Configure MCP servers:
   Update `config/profiles.yaml` with your MCP server configurations.

## Usage

1. Start the bot:
   ```bash
   python telegram_bot.py
   ```

2. In Telegram:
   - Start a chat with your bot
   - Send any query
   - The bot will process your request and respond

## Project Structure

```
S8MCPGoesToInternet/
├── config/
│   └── profiles.yaml
├── core/
│   ├── loop.py      # Main agent processing loop
│   ├── session.py   # MCP server connection management
│   ├── context.py   # Agent state management
│   └── strategy.py  # Decision-making logic
├── modules/
│   ├── perception.py    # Input analysis
│   ├── memory.py        # Memory management
│   ├── action.py        # Tool execution
│   ├── decision.py      # Decision making
│   ├── tools.py         # Tool management
│   └── model_manager.py # LLM interaction
├── mcp/
├── .env
├── .gitignore
├── agent.py
├── main.py
├── pyproject.toml
├── README.md
├── telegram_bot.py
└── token.json
```

## Dependencies

Key dependencies include:
- `pytelegrambotapi`: Telegram bot integration
- `mcp`: Model Context Protocol support
- `google-genai`: Google's Generative AI integration
- `llama-index`: Document processing and indexing
- `fastapi`: API framework
- Various utility libraries for enhanced functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Support

For support, please [add your support contact information]
