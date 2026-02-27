# Session 4: MCP, Plugins & Marketplaces
## Cut the Crap — AI Engineer Edition

---

## Slide 1: The Problem MCP Solves (Topic 16)

**SHOW:**
```
Before MCP:
┌──────────┐    custom code    ┌──────────┐
│  Claude   │◄────────────────►│  GitHub   │
└──────────┘                   └──────────┘
┌──────────┐    custom code    ┌──────────┐
│  GPT-5.2   │◄────────────────►│  GitHub   │
└──────────┘                   └──────────┘
┌──────────┐    custom code    ┌──────────┐
│  Gemini   │◄────────────────►│  GitHub   │
└──────────┘                   └──────────┘

N models × M tools = N×M integrations  😱

After MCP:
┌──────────┐                   ┌──────────┐
│  Claude   │◄──┐              │  GitHub   │◄── MCP Server
└──────────┘   │    ┌──────┐  └──────────┘
┌──────────┐   ├───►│ MCP  │  ┌──────────┐
│  GPT-5.2   │◄──┤   │Protocol│►│  Slack    │◄── MCP Server
└──────────┘   │    └──────┘  └──────────┘
┌──────────┐   │              ┌──────────┐
│  Gemini   │◄──┘              │  Database │◄── MCP Server
└──────────┘                   └──────────┘

N + M integrations  ✅
```

**SAY:**
> MCP — Model Context Protocol — is an open standard from Anthropic that's been adopted across the industry. The problem: if you have 3 AI models and 10 tools, you need 30 custom integrations. MCP creates a universal plug — any MCP server works with any MCP client. Build the GitHub integration once, it works everywhere. It's USB-C for AI tools.

---

## Slide 2: MCP Architecture

**SHOW:**
```
┌─────────────────────────────────────────────────┐
│                  MCP HOST                        │
│  (Claude Desktop, OpenClaw, VS Code, your app)  │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐              │
│  │  MCP Client  │  │  MCP Client  │  ...         │
│  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼─────────────────────┘
          │ stdio/SSE        │ stdio/SSE
          ▼                  ▼
   ┌──────────────┐  ┌──────────────┐
   │  MCP Server  │  │  MCP Server  │
   │  (GitHub)    │  │  (Postgres)  │
   └──────────────┘  └──────────────┘

MCP Server exposes:
  📋 Tools      — functions the AI can call
  📄 Resources  — data the AI can read (files, DB records)
  💬 Prompts    — reusable prompt templates

Transport:
  stdio  — local process (most common)
  SSE    — remote over HTTP
```

**SAY:**
> Here's the architecture. The Host is your application — Claude Desktop, OpenClaw, VS Code with Copilot. Inside it, MCP Clients connect to MCP Servers. Each server exposes tools, resources, and prompts. Transport is usually stdio for local servers — the host spawns a process and communicates over stdin/stdout. For remote servers, it's SSE over HTTP. The key insight: the server is just a process that speaks a JSON-RPC protocol.

---

## Slide 3: What MCP Servers Exist (Topic 17)

**SHOW:**
```
Popular MCP Servers (as of 2026):

Filesystem & Code:
  📁 filesystem    — read/write/search files
  🔧 git           — clone, diff, commit, log
  💻 github        — issues, PRs, repos, actions

Data:
  🐘 postgres      — query PostgreSQL databases
  📊 sqlite        — local SQLite databases
  🔍 elasticsearch — search and analytics

Communication:
  💬 slack          — channels, messages, users
  📧 gmail         — read/send email
  📝 notion        — pages, databases

Web:
  🌐 brave-search  — web search
  🕷️ puppeteer     — browser automation
  📡 fetch         — HTTP requests

Dev Tools:
  🐳 docker        — container management
  ☁️  aws           — AWS service access
  📦 npm           — package info
```

**SAY:**
> The ecosystem is huge. Filesystem gives AI read/write access to your files. GitHub lets it manage issues and PRs. Postgres lets it query your database directly. Slack lets it read and send messages. These aren't toy demos — production teams use these daily. The community maintains hundreds of servers. If one doesn't exist for your tool, building one is straightforward.

---

## Slide 4: Setting Up MCP — Claude Desktop (Topic 18)

**SHOW:**
```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
// (macOS) or %APPDATA%/Claude/claude_desktop_config.json (Windows)

{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/projects"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://user:pass@localhost:5432/mydb"
      ]
    }
  }
}
```

```
Setup steps:
1. Install Node.js (npx comes with it)
2. Edit the config file above
3. Restart Claude Desktop
4. Look for 🔌 icon — tools appear automatically
5. Ask Claude: "List the files in my projects folder"
```

**SAY:**
> Live demo time. This is the Claude Desktop config file. You add MCP servers here and restart. Each server has a command to run and optional environment variables. The filesystem server needs a path to expose. GitHub needs a token. Postgres needs a connection string. After restart, Claude Desktop shows a plug icon — click it to see available tools. Then just ask Claude to do things and it calls the tools automatically. Let's set this up right now.

---

## Slide 5: Setting Up MCP — VS Code / Cursor

**SHOW:**
```json
// .vscode/mcp.json (project-level)
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${env:GITHUB_TOKEN}"
      }
    }
  }
}
```

```
// For Cursor: similar config in Cursor settings
// For Claude Code CLI:
// ~/.claude/settings.json or project-level .mcp.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

**SAY:**
> VS Code and Cursor also support MCP. Put a `mcp.json` in your `.vscode` folder and the AI assistant gets tool access. Cursor has similar support. Claude Code CLI reads from `.mcp.json` in your project. The config format is nearly identical everywhere — that's the point of a standard protocol.

---

## Slide 6: Building a Simple MCP Server

**SHOW:**
```python
# my_mcp_server.py — A minimal MCP server in Python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("my-tools")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="word_count",
            description="Count words in a text",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to count words in"}
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="reverse_string",
            description="Reverse a string",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "word_count":
        count = len(arguments["text"].split())
        return [TextContent(type="text", text=f"Word count: {count}")]
    elif name == "reverse_string":
        return [TextContent(type="text", text=arguments["text"][::-1])]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```bash
# Install the MCP Python SDK
pip install mcp

# Register in Claude Desktop config:
# "my-tools": {"command": "python", "args": ["my_mcp_server.py"]}
```

**SAY:**
> Building an MCP server is surprisingly simple. The Python SDK gives you a Server class. You decorate functions to list tools and handle calls. The server communicates over stdio — Claude Desktop spawns it as a process. This is maybe 40 lines of real code. You could wrap any internal API, any database, any service as an MCP server in under an hour.

---

## Slide 7: MCP Resources & Prompts

**SHOW:**
```python
from mcp.types import Resource, Prompt, PromptMessage, PromptArgument

# Resources — data the AI can read
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="config://app-settings",
            name="App Settings",
            description="Current application configuration",
            mimeType="application/json",
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "config://app-settings":
        return json.dumps({"debug": True, "version": "2.1.0"})

# Prompts — reusable prompt templates
@server.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="code-review",
            description="Review code for bugs and improvements",
            arguments=[
                PromptArgument(name="language", description="Programming language"),
                PromptArgument(name="code", description="Code to review"),
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "code-review":
        return {
            "messages": [
                PromptMessage(role="user", content=TextContent(
                    type="text",
                    text=f"Review this {arguments['language']} code for bugs, "
                         f"security issues, and improvements:\n\n{arguments['code']}"
                ))
            ]
        }
```

**SAY:**
> MCP isn't just tools. Resources let the AI read data — config files, database records, API state. The AI can browse available resources and read what it needs. Prompts are reusable templates — think of them as saved prompt recipes the AI can use. In practice, tools are used 90% of the time, but resources and prompts round out the protocol.

---

## Slide 8: Marketplaces (Topic 19)

**SHOW:**
```
GPT Store (OpenAI):
  - Largest marketplace
  - Custom GPTs built by anyone
  - Revenue sharing for creators
  - Quality varies wildly
  - Accessible from ChatGPT

ClawHub (OpenClaw):
  - Skills marketplace for OpenClaw agents
  - Code-based — more powerful than GPTs
  - MCP server integration
  - Community-driven
  - Growing ecosystem

Community MCP Servers:
  - github.com/modelcontextprotocol/servers (official)
  - github.com/punkpeye/awesome-mcp-servers (community list)
  - npm/PyPI packages — install and configure
  - No centralized "store" yet — it's early

Smithery.ai:
  - MCP server directory and registry
  - One-click install for supported hosts
  - Growing catalog
```

**SAY:**
> Marketplaces are still early. The GPT Store is the biggest but quality is all over the place. ClawHub is where OpenClaw skills live — we'll build one in Session 8. For MCP, there's no single store yet — you find servers on GitHub, npm, and directories like Smithery. The official MCP GitHub has reference servers. The awesome-mcp-servers list is community-curated. This space will consolidate over the next year.

---

## Slide 9: Hands-On — Connect MCP (Topic 20)

**SHOW:**
```
📝 Exercise: Set up MCP in Claude Desktop (or OpenClaw)

Part 1 — Connect existing servers (10 min):
  1. Add filesystem MCP server to Claude Desktop
  2. Add GitHub MCP server (create a token first)
  3. Ask Claude to list files, read a file, create a file
  4. Ask Claude to list your GitHub repos

Part 2 — Build your own server (20 min):
  1. Create a simple MCP server with 2-3 custom tools
     Ideas: todo list, dictionary lookup, unit converter
  2. Register it in Claude Desktop config
  3. Test it by chatting with Claude

Bonus:
  - Set up the Postgres MCP server with a local database
  - Ask Claude to write and run SQL queries
```

**SAY:**
> Two-part exercise. First, connect the official filesystem and GitHub MCP servers to Claude Desktop. This should take 10 minutes — it's just config. Then build your own MCP server with custom tools. Use the template from slide 6. Register it, restart Claude Desktop, and test it. If you finish early, try the Postgres server — there's nothing quite like asking Claude to query your database in natural language.

---

## Slide 10: MCP Security Considerations

**SHOW:**
```
⚠️ MCP Security — Think Before You Connect

Filesystem Server:
  - Only expose directories you intend to
  - AI CAN write/delete files if the server allows it
  - Use read-only mode when possible

Database Servers:
  - Use read-only database users when possible
  - Never connect to production DBs without safeguards
  - Review queries before execution (some clients show them)

GitHub/Slack/Email:
  - Use tokens with minimum necessary permissions
  - The AI can send messages, create issues, merge PRs
  - Audit what actions the AI takes

General:
  ✅ Principle of least privilege
  ✅ Review tool calls before they execute
  ✅ Use sandboxed/dev environments first
  ✅ Log everything
  ❌ Don't give AI access to production systems without approval flows
```

**SAY:**
> Quick but critical: MCP gives AI real power. That filesystem server can delete files. That GitHub server can merge PRs. That database server can run DELETE queries. Principle of least privilege — expose only what's needed, use read-only tokens where possible, and always test in dev first. Most MCP clients show you what the AI wants to do before executing it. Pay attention to those prompts.

---

## Slide 11: Session 4 Recap

**SHOW:**
```
✅ MCP = USB-C for AI tools — universal, open standard
✅ Architecture: Host → Client → Server (stdio or SSE)
✅ Servers expose: Tools, Resources, Prompts
✅ Setup: JSON config in Claude Desktop / VS Code / OpenClaw
✅ Building servers: ~40 lines with Python SDK
✅ Ecosystem: 100s of servers on GitHub, Smithery, npm
✅ Marketplaces: GPT Store, ClawHub, community MCP
✅ Security: least privilege, audit, sandbox first

Sessions 1-4 complete! You now know:
  → The landscape & APIs
  → Prompt engineering & structured output
  → Tool use & function calling
  → MCP & the tool ecosystem

Next half: Agents, RAG, Evals, Production
```

**SAY:**
> That's the first half of the course done. You've gone from making your first API call to building tool-calling assistants to connecting universal tool protocols. Sessions 5-8 build on everything: agents that chain multiple tools together, RAG for working with your own data, evals and observability for production quality, and finally shipping real AI applications. See you in Session 5.
