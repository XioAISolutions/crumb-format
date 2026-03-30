# CRUMB Launch Posts

---

## 1. Show HN

**Title:** Show HN: CRUMB – A copy-paste format for handing off context between AI tools

**Body:**

I kept running into the same problem: I'd spend 30 minutes getting Claude to understand my codebase, architecture decisions, and current task — then need to switch to Cursor or ChatGPT for a different strength, and have to re-explain everything from scratch.

There's no standard way to move working context between AI tools. So I built CRUMB — a structured plain-text format that captures what you're working on, what's been done, what's left, key decisions, and relevant file contents. You paste it into the next tool and it picks up where you left off.

It's deliberately low-tech. No API integrations, no accounts, no vendor lock-in. Just a structured text block you copy and paste. The format is human-readable so you can edit it by hand if you want.

What's included:

- **Web converter** — paste unstructured notes, get a CRUMB block back: https://xioaisolutions.github.io/crumb-format
- **CLI** with 24 commands (`crumb create`, `crumb convert`, `crumb diff`, `crumb lint`, etc.)
- **MCP server** for Claude Desktop and compatible clients
- **Browser extension** for one-click capture
- **Cursor rules** and **CLAUDE.md** integrations so your editor AI understands the format natively

Install: `pip install crumb-format`

GitHub: https://github.com/XioAISolutions/crumb-format

I'd genuinely like feedback on the format spec itself. Is it capturing the right information? Too verbose? Missing sections that would help the receiving AI get up to speed faster?

---

## 2. Reddit r/ChatGPT

**Title:** I got mass of losing all my context every time I switch between ChatGPT and another AI, so I made a copy-paste handoff format

**Body:**

Does anyone else do this?

1. Spend 20 minutes getting ChatGPT to fully understand your project
2. Realize you need Claude or Cursor for a specific task
3. Open new chat, stare at empty text box, try to remember everything you just explained
4. Give up and re-explain half of it, forget the other half
5. The new AI confidently goes in the wrong direction because it's missing context

I got tired of this loop and built CRUMB — it's basically a structured text format for AI-to-AI handoffs. You (or the AI) fill in a template with the project state, what's done, what's next, key decisions, and relevant code. Then you paste it into the next tool and it actually knows what's going on.

**Fastest way to try it:** Paste your messy project notes into the web converter and it'll give you a clean CRUMB block: https://xioaisolutions.github.io/crumb-format

No sign-up, no install needed for that. If you want the full CLI with 24 commands, it's `pip install crumb-format`.

It works with ChatGPT, Claude, Cursor, Gemini, Copilot — anything that accepts text, really. That's the whole point. It's just a structured text block, not a platform.

GitHub if you want to look under the hood: https://github.com/XioAISolutions/crumb-format

Curious if other people have been dealing with this same context-loss problem or if I'm the only one who switches tools mid-task constantly.

---

## 3. Reddit r/cursor

**Title:** Built a structured handoff format for moving context between Cursor and other AI tools — includes Cursor rules integration

**Body:**

If you're like me, you use Cursor as your main editor but still jump into Claude or ChatGPT for planning, debugging help, or getting a second opinion on architecture. The problem is that every time you switch, you lose all the context Cursor had about your project.

CRUMB is a plain-text format designed to capture working context — project state, completed work, remaining tasks, decisions made, relevant code — in a way that any AI tool can parse immediately.

**Cursor-specific stuff:**

- `crumb init --cursor-rules` generates a `.cursorrules` file that teaches Cursor how to read and write CRUMB blocks natively. Once it's there, you can ask Cursor to "generate a CRUMB handoff for this session" and it'll output a properly formatted block.
- The MCP server (`crumb mcp-serve`) works with any MCP-compatible client, so if you're using Claude Desktop alongside Cursor, both tools can read/write CRUMBs through the same interface.
- `crumb create --from-git` can auto-generate a handoff from your recent git activity, which is useful when you want to hand off a half-finished branch to another tool or collaborator.

**Quick start:**

```
pip install crumb-format
cd your-project
crumb init --cursor-rules
```

Or skip the install entirely and try the web converter: https://xioaisolutions.github.io/crumb-format

There are 24 CLI commands total — `crumb lint` validates your blocks, `crumb diff` compares two handoffs, `crumb convert` moves between formats, etc.

GitHub: https://github.com/XioAISolutions/crumb-format

Would be interested to hear how other Cursor users handle the context-switching problem. Do you just keep everything in Cursor, or do you bounce between tools too?

---

## 4. Reddit r/ClaudeAI

**Title:** Made a structured context-handoff format with CLAUDE.md integration — so Claude remembers your project state across tools

**Body:**

Claude is great at understanding complex project context — but that context lives and dies with a single conversation. If you need to continue in Cursor, ChatGPT, or even a new Claude chat, you're starting over.

CRUMB is a plain-text handoff format that solves this. It captures the working state of your project — what you're building, what's done, what's left, key decisions, relevant code — in a structured block you paste into whatever tool comes next.

**Claude-specific features:**

- `crumb init --claude-md` generates a `CLAUDE.md` file that teaches Claude how to read, write, and request CRUMB blocks. Claude will understand the format in context and can generate handoffs on request.
- The MCP server (`crumb mcp-serve`) integrates with Claude Desktop directly. Claude can create, validate, and manage CRUMB blocks through tool use — no copy-paste needed in that workflow.
- Works well with Claude Projects: include the CRUMB spec in your project knowledge, and every conversation in that project can produce and consume standardized handoffs.

**Try it right now** — paste your project notes into the web converter: https://xioaisolutions.github.io/crumb-format

Or install locally:

```
pip install crumb-format
crumb init --claude-md
```

24 CLI commands for creating, validating, diffing, and converting CRUMB blocks. Full details on GitHub: https://github.com/XioAISolutions/crumb-format

For anyone who uses Claude alongside other AI tools — how do you currently handle the context transfer? Just curious what workarounds people have cobbled together.

---

## 5. X/Twitter Thread

**Tweet 1 (Hook):**

I switch between Claude, Cursor, and ChatGPT constantly.

The worst part isn't picking the right tool — it's re-explaining my entire project every single time I switch.

So I built a fix. It's called CRUMB. Thread:

**Tweet 2 (Problem):**

The problem is simple: AI tools don't talk to each other.

You spend 30 min getting one AI to understand your codebase. Then you switch tools and start from zero.

Context is the most expensive thing in AI-assisted work, and we keep throwing it away.

**Tweet 3 (Solution):**

CRUMB is a structured text format for AI handoffs.

It captures: project state, completed work, remaining tasks, decisions made, and relevant code.

You paste it into the next tool. The AI picks up where the last one left off.

No APIs. No accounts. Just text.

**Tweet 4 (What shipped):**

Just went live with:

- Web converter (try it instantly)
- pip install crumb-format
- 24 CLI commands
- MCP server for Claude Desktop
- Browser extension
- Cursor rules + CLAUDE.md integrations

**Tweet 5 (CTA):**

Try it in 10 seconds — paste your project notes here and get a structured handoff block back:

https://xioaisolutions.github.io/crumb-format

GitHub: https://github.com/XioAISolutions/crumb-format

`pip install crumb-format`

Would love feedback on the format itself. What context do you wish transferred between AI tools?

---

## 6. LinkedIn Post

I've been building with AI tools every day for the past year — Claude, Cursor, ChatGPT, Gemini, Copilot. Each has strengths that make it worth reaching for at different stages of a task.

But there's a gap nobody's really addressed: context doesn't transfer between them.

You spend real time getting one AI up to speed on your project — the architecture, the constraints, what you've tried, what didn't work, what's left to do. Then you switch tools and all of that evaporates. You start over. Every time.

Multiply that across a team where different people prefer different AI tools, and the inefficiency compounds fast.

So I built CRUMB — an open-source, plain-text format for structured AI handoffs.

It's deliberately simple. A CRUMB block captures the working state of a task: what the project is, what's been done, what's remaining, key decisions and their rationale, and relevant code or file contents. You paste it into whatever tool comes next, and it has the full picture.

No platform lock-in. No API integrations required. No accounts. It's just structured text — the most portable format there is.

What shipped today:

- A web converter where you can try it instantly (link in comments)
- A CLI with 24 commands: pip install crumb-format
- An MCP server for Claude Desktop integration
- A browser extension
- Native integrations for Cursor and Claude via config files

The whole thing is open source: https://github.com/XioAISolutions/crumb-format

Try the web converter here: https://xioaisolutions.github.io/crumb-format

I'm building this in public and would genuinely appreciate feedback — especially from teams that use multiple AI tools in their workflow. What context matters most when you switch between them?

#OpenSource #AI #DeveloperTools #BuildingInPublic

---

## 7. Dev.to / Blog Post Outline

**Title:** "I Built a Copy-Paste Format for AI Tool Handoffs (and Why Context Is the Real Bottleneck)"

### Section 1: The Problem Nobody Talks About

- AI tools don't interoperate; context is trapped in individual conversations
- The real cost isn't subscription fees — it's the time spent re-explaining projects
- Each tool has strengths (Claude for reasoning, Cursor for code editing, ChatGPT for broad knowledge), so switching is inevitable
- The "start from scratch" tax adds up to hours per week for heavy AI users

### Section 2: Why Existing Solutions Fall Short

- Chat export is unstructured and full of noise
- System prompts help one tool but don't transfer
- Custom GPTs / Claude Projects hold knowledge but not session state
- API-based solutions create vendor lock-in and add complexity

### Section 3: Introducing CRUMB — A Plain-Text Handoff Format

- What a CRUMB block looks like (annotated example)
- Design principles: human-readable, AI-parseable, tool-agnostic, copy-paste friendly
- The section structure: metadata, project overview, completed work, remaining tasks, decisions, context files
- Why plain text wins over JSON/YAML for this use case (paste it anywhere)

### Section 4: The Tooling Ecosystem

- **Web converter** — paste messy notes, get a clean CRUMB block: https://xioaisolutions.github.io/crumb-format
- **CLI** (`pip install crumb-format`) — 24 commands for power users
  - `crumb create` — interactive or from flags
  - `crumb convert` — transform between formats
  - `crumb lint` — validate structure
  - `crumb diff` — compare two handoffs
  - `crumb init --cursor-rules` / `crumb init --claude-md` — editor integrations
- **MCP server** — native integration with Claude Desktop and compatible clients
- **Browser extension** — capture context from any web page

### Section 5: Real Workflow Examples

- Example 1: Planning in ChatGPT, implementing in Cursor, debugging with Claude
- Example 2: Handing off a half-finished feature to a teammate who uses different tools
- Example 3: Resuming your own work after a context window fills up
- Show the actual CRUMB block at each handoff point

### Section 6: Design Decisions and Trade-offs

- Why sections are ordered the way they are (most important context first)
- Why metadata is minimal (avoid stale information)
- Why code snippets are included inline instead of referenced (portability over brevity)
- What didn't work in earlier versions of the format

### Section 7: Try It Yourself

- Quickstart in 30 seconds with the web converter
- Full install: `pip install crumb-format`
- GitHub: https://github.com/XioAISolutions/crumb-format
- What feedback would be most useful (format coverage, missing sections, verbosity)

### Section 8: What's Next

- Community-contributed templates for common workflows
- IDE plugins beyond Cursor
- Format versioning and backward compatibility
- The bigger picture: AI tools need interoperability standards, and it starts with context
