# MCP Wrapper

`snipara-memory` ships with a local MCP stdio server so MCP-compatible clients
can talk to the standalone engine directly.

## Run

```bash
snipara-memory mcp
```

Use an explicit store file if you want a repo-local memory corpus:

```bash
snipara-memory mcp --store-path ./.snipara-memory.json
```

## Exposed Tools

- `memory_store`
- `memory_recall`
- `memory_session_bundle`
- `memory_list`
- `memory_detect_contradictions`
- `memory_resolve_contradiction`
- `memory_import_transcript`
- `memory_import_project`

## Example Claude Desktop Config

```json
{
  "mcpServers": {
    "snipara-memory": {
      "command": "uvx",
      "args": ["snipara-memory", "mcp", "--store-path", "/absolute/path/.snipara-memory.json"]
    }
  }
}
```

## Scope

This MCP wrapper is local-first and standalone.

It is not a hosted Snipara Cloud transport. It exists so the OSS memory engine
can be automated and integrated with MCP-native tools.
