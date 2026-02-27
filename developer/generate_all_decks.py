"""Generate all 8 developer PowerPoint decks."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import copy

# Colors
NAVY = RGBColor(0x1B, 0x2A, 0x4A)
ACCENT = RGBColor(0x00, 0x96, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
MED_GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF0, 0xF0, 0xF0)
CODE_BG = RGBColor(0xF0, 0xF0, 0xF0)
TABLE_HEADER = RGBColor(0x1B, 0x2A, 0x4A)
TABLE_ALT = RGBColor(0xF5, 0xF7, 0xFA)


def new_prs():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_title_slide(prs, title, subtitle, session_num):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Navy background
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = NAVY

    # Title
    left, top, w, h = Inches(1), Inches(2.0), Inches(11.333), Inches(1.5)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT

    # Subtitle
    left, top, w, h = Inches(1), Inches(3.6), Inches(11.333), Inches(1.0)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(22)
    p.font.color.rgb = ACCENT
    p.alignment = PP_ALIGN.LEFT

    # Session number
    left, top, w, h = Inches(1), Inches(5.0), Inches(11.333), Inches(0.6)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = f"Session {session_num} of 8  •  Cut the Crap — AI Engineer Edition  •  February 2026"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(0x88, 0x99, 0xBB)
    p.alignment = PP_ALIGN.LEFT

    # Notes
    slide.notes_slide.notes_text_frame.text = f"Welcome to Session {session_num}. {subtitle}"
    return slide


def add_content_slide(prs, title, bullets, notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # Accent bar at top
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.08))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()

    # Title
    left, top, w, h = Inches(0.8), Inches(0.3), Inches(11.733), Inches(0.9)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.LEFT

    # Bullets
    left, top, w, h = Inches(1.0), Inches(1.4), Inches(11.333), Inches(5.5)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        
        # Handle sub-bullets (indented with "  - ")
        if bullet.startswith("  - "):
            p.text = bullet.strip("- ").strip()
            p.level = 1
            p.font.size = Pt(18)
            p.font.color.rgb = MED_GRAY
        else:
            p.text = bullet
            p.level = 0
            p.font.size = Pt(20)
            p.font.color.rgb = DARK_GRAY
        
        p.space_after = Pt(8)
        p.space_before = Pt(4)

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_code_slide(prs, title, code, language="python", notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Accent bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.08))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()

    # Title
    left, top, w, h = Inches(0.8), Inches(0.3), Inches(11.733), Inches(0.8)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NAVY

    # Code box with background
    left, top, w, h = Inches(0.8), Inches(1.3), Inches(11.733), Inches(5.7)
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = CODE_BG
    shape.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    shape.line.width = Pt(1)

    # Language label
    left2, top2, w2, h2 = Inches(0.9), Inches(1.35), Inches(2), Inches(0.35)
    txBox2 = slide.shapes.add_textbox(left2, top2, w2, h2)
    tf2 = txBox2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = language
    p2.font.size = Pt(11)
    p2.font.color.rgb = MED_GRAY
    p2.font.bold = True

    # Code text
    left, top, w, h = Inches(1.0), Inches(1.7), Inches(11.333), Inches(5.2)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = code
    p.font.size = Pt(13)
    p.font.name = "Consolas"
    p.font.color.rgb = DARK_GRAY
    p.alignment = PP_ALIGN.LEFT

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_table_slide(prs, title, headers, rows, notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Accent bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.08))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()

    # Title
    left, top, w, h = Inches(0.8), Inches(0.3), Inches(11.733), Inches(0.8)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = NAVY

    # Table
    num_rows = len(rows) + 1
    num_cols = len(headers)
    left, top, w, h = Inches(0.8), Inches(1.4), Inches(11.733), Inches(0.4 * num_rows)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, w, h)
    table = table_shape.table

    # Set column widths evenly
    col_width = int(Inches(11.733) / num_cols)
    for i in range(num_cols):
        table.columns[i].width = col_width

    # Header row
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = TABLE_HEADER
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    # Data rows
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = str(val)
            # Alternating row colors
            if r % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TABLE_ALT
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(13)
                p.font.color.rgb = DARK_GRAY
                p.alignment = PP_ALIGN.CENTER

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_two_column_slide(prs, title, left_title, left_bullets, right_title, right_bullets, notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Accent bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.08))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()

    # Title
    left, top, w, h = Inches(0.8), Inches(0.3), Inches(11.733), Inches(0.8)
    txBox = slide.shapes.add_textbox(left, top, w, h)
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = NAVY

    # Left column header
    left_x = Inches(0.8)
    txBox = slide.shapes.add_textbox(left_x, Inches(1.3), Inches(5.5), Inches(0.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = left_title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    # Left bullets
    txBox = slide.shapes.add_textbox(left_x, Inches(1.9), Inches(5.5), Inches(5.0))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, b in enumerate(left_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = b
        p.font.size = Pt(18)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(6)

    # Right column header
    right_x = Inches(7.0)
    txBox = slide.shapes.add_textbox(right_x, Inches(1.3), Inches(5.5), Inches(0.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = right_title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    # Right bullets
    txBox = slide.shapes.add_textbox(right_x, Inches(1.9), Inches(5.5), Inches(5.0))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, b in enumerate(right_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = b
        p.font.size = Pt(18)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(6)

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


# ============================================================
# SESSION 1
# ============================================================
def build_session_1():
    prs = new_prs()
    
    add_title_slide(prs, "The AI Landscape & APIs", "Models, providers, and your first API call", 1)

    add_content_slide(prs, "The Big Three + Challengers", [
        "OpenAI — GPT-5.2, GPT-4.1, o3/o4-mini reasoning models",
        "Anthropic — Opus 4.6, Sonnet 4.6, Haiku 3.5 (Claude Code)",
        "Google — Gemini 2.5 Pro, Gemini 2.5 Flash",
        "Challengers: DeepSeek V4, Meta Llama 4, Grok 4, Mistral Large",
        "All accessible through simple REST APIs + SDKs",
    ], "Overview of the current AI provider landscape. Emphasize that the API patterns are nearly identical across providers.")

    add_table_slide(prs, "Model Comparison — February 2026",
        ["Provider", "Flagship", "Mid-Tier", "Fast/Cheap", "Pricing (In/Out per 1M)"],
        [
            ["OpenAI", "GPT-5.2", "GPT-4.1", "GPT-4.1-mini / nano", "$20 / $60"],
            ["Anthropic", "Opus 4.6", "Sonnet 4.6", "Haiku 3.5", "$5 / $25"],
            ["Google", "Gemini 2.5 Pro", "Gemini 2.5 Flash", "—", "Varies"],
            ["DeepSeek", "V4", "—", "—", "$0.14 / 1M"],
            ["Meta", "Llama 4", "—", "—", "Free (self-host)"],
        ],
        "Walk through pricing. DeepSeek V4 is absurdly cheap. Llama 4 is free to self-host but you pay for compute.")

    add_two_column_slide(prs, "Reasoning vs Non-Reasoning Models",
        "Reasoning Models", [
            "GPT-5.2 / o3 / o4-mini",
            "Think before answering (chain-of-thought)",
            "Best for math, logic, complex analysis",
            "Higher cost, higher latency",
            "Temperature usually fixed or low",
        ],
        "Non-Reasoning Models", [
            "GPT-4.1 / Sonnet 4.6 / Gemini 2.5 Flash",
            "Direct response, no thinking phase",
            "Best for chat, code, writing, tool use",
            "Cheaper, faster",
            "Adjustable temperature (0.0–2.0)",
        ],
        "Key distinction students need to understand. Reasoning models 'think' internally. Non-reasoning models are faster and cheaper for most tasks.")

    add_content_slide(prs, "Open Source & Self-Hosting", [
        "Llama 4 (Meta) — strongest open model, multiple sizes",
        "DeepSeek V4 — insanely cheap via API ($0.14/1M tokens)",
        "Mistral Large — strong European alternative",
        "Qwen 2.5 — Alibaba, popular in Asia",
        "Self-hosting: Ollama (easy local), vLLM (production GPU serving)",
        "Trade-off: control & privacy vs. effort & cost of GPUs",
    ], "Open source is viable for many use cases. Ollama makes local dev easy. vLLM for production serving.")

    add_two_column_slide(prs, "OAuth vs API Key — How Authentication Works",
        "OAuth (Subscription-Based)", [
            "App gives you a link → open in browser",
            "Authorize with your existing account",
            "Paste the code back into the app",
            "Uses your existing subscription (Pro/Team)",
            "Supported by: OpenClaw, Claude Code",
            "No credit card / billing setup needed",
        ],
        "API Key (Pay-as-You-Go)", [
            "Go to provider console → create key",
            "Add credits ($5+ to start)",
            "Pay per token used",
            "Required for custom programs",
            "Full programmatic control",
            "Your code ONLY uses API keys",
        ],
        "Critical distinction. OAuth = easy, uses existing sub. API key = for developers building custom apps. OpenClaw accepts BOTH — you choose during 'openclaw setup'.")

    add_content_slide(prs, "Setting Up API Keys (Live Demo)", [
        "OpenAI: platform.openai.com → API keys → Create → Add credits",
        "Anthropic: console.anthropic.com → API Keys → Create → Add credits",
        "Google: aistudio.google.com → Get API key (free tier available)",
        "Store in environment: export OPENAI_API_KEY=sk-...",
        "Never commit keys to git — use .env files",
        "Start with $5–10 of credits, that's plenty for learning",
    ], "Live demo: create keys on each platform. Show the billing/credits page. Emphasize .env files and .gitignore.")

    add_content_slide(prs, "SDK Installation & First Call", [
        "pip install openai anthropic google-genai",
        "All three SDKs follow the same pattern:",
        "  - Create a client (reads API key from env)",
        "  - Build a messages array (system + user + assistant)",
        "  - Call the model, get a response",
        "Let's look at the code →",
    ], "Show the SDK install command. Explain the universal message format that all providers use.")

    add_code_slide(prs, "OpenAI — Your First API Call", """from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"}
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)""", "python",
        "Simplest possible OpenAI call. Point out: client auto-reads env var, messages array, model name, temperature.")

    add_code_slide(prs, "Anthropic — Same Pattern, Different SDK", """import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

response = client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=8192,          # Required for Anthropic
    system="You are a helpful assistant.",  # System is separate
    messages=[
        {"role": "user", "content": "What is AI?"}
    ],
)
print(response.content[0].text)""", "python",
        "Key differences: max_tokens is required, system prompt is a separate parameter, response structure differs.")

    add_code_slide(prs, "Google Gemini — Third Provider, Same Idea", """from google import genai
client = genai.Client()  # reads GOOGLE_API_KEY

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is AI?",
    config={"system_instruction": "You are a helpful assistant."},
)
print(response.text)""", "python",
        "Gemini SDK is slightly different but same concept. Contents can be a simple string or list of parts.")

    add_content_slide(prs, "Streaming Responses", [
        "Without streaming: wait for full response, then display",
        "With streaming: tokens arrive one at a time, display instantly",
        "Much better UX — user sees response forming in real-time",
        "OpenAI: stream=True → iterate over chunks",
        "Anthropic: client.messages.stream() context manager",
        "Google: generate_content_stream() → iterate",
    ], "Streaming is essential for any chat UI. Show the streaming variants from chat_script.py.")

    add_code_slide(prs, "Streaming — OpenAI Example", """stream = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)""", "python",
        "Key pattern: stream=True, then iterate. Each chunk has a delta with partial content.")

    add_content_slide(prs, "Multi-Provider Chat Script (Hands-On)", [
        "chat_script.py — switch providers mid-conversation",
        "Commands: /openai  /anthropic  /google  /clear  /quit",
        "Shared message history across providers",
        "Streaming responses from all three",
        "Exercise: run it, try each provider, compare responses",
        "Notice: same question → different styles & quality",
    ], "This is the hands-on exercise. Students run chat_script.py and experiment with all 3 providers.")

    add_content_slide(prs, "Key Takeaways — Session 1", [
        "Three major providers + strong open-source alternatives",
        "Reasoning models think harder; non-reasoning are faster/cheaper",
        "OAuth for apps like OpenClaw; API keys for your custom code",
        "All SDKs follow client → messages → response pattern",
        "Streaming is essential for good UX",
        "Next session: prompt engineering & multimodal APIs",
    ], "Recap the session. Preview Session 2.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-1/session-1-landscape-apis.pptx")
    return len(prs.slides)


# ============================================================
# SESSION 2
# ============================================================
def build_session_2():
    prs = new_prs()

    add_title_slide(prs, "Prompting, Structured Output\n& Multimodal APIs", "System prompts, JSON schemas, vision, audio, and image generation", 2)

    add_content_slide(prs, "Prompt Engineering Fundamentals", [
        "System prompt: sets persona, rules, and constraints",
        "Few-shot examples: show the model what you want",
        "Chain-of-thought: 'Think step by step' improves reasoning",
        "Be specific: 'in 3 bullet points' beats 'briefly'",
        "Negative instructions work: 'Do NOT include disclaimers'",
        "Temperature: 0 = deterministic, 1+ = creative",
    ], "Core prompting techniques. These work across all providers. Temperature is the most important parameter after the prompt itself.")

    add_content_slide(prs, "System Prompt Best Practices", [
        "Define role: 'You are a senior Python developer'",
        "Set format: 'Respond in JSON with keys: answer, confidence'",
        "Add constraints: 'Maximum 100 words. No markdown.'",
        "Include examples in the system prompt for consistency",
        "Anthropic: system is a separate parameter (not in messages)",
        "Test with edge cases — adversarial inputs reveal weaknesses",
    ], "System prompts are your primary control mechanism. Show examples of good vs bad system prompts.")

    add_two_column_slide(prs, "Structured Output — Three Approaches",
        "OpenAI (Native)", [
            "response_format=YourPydanticModel",
            "Guaranteed valid JSON matching schema",
            "Works with GPT-4.1, GPT-5.2",
            "Best: client.beta.chat.completions.parse()",
            "Returns .parsed with typed object",
        ],
        "Anthropic (Tool-Use Trick)", [
            "Define a 'tool' with your schema",
            "Force tool_choice to that tool name",
            "Model outputs structured data as tool input",
            "Not 'real' tool use — just schema enforcement",
            "Works reliably with Sonnet 4.6+",
        ],
        "OpenAI has the cleanest structured output. Anthropic uses the tool-use trick. Both work well.")

    add_code_slide(prs, "OpenAI Structured Output", """from pydantic import BaseModel, Field

class ImageAnalysis(BaseModel):
    description: str = Field(description="One paragraph description")
    mood: str = Field(description="happy, sad, neutral, etc.")
    objects: list[str] = Field(description="Objects detected")

response = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Analyze this photo..."}],
    response_format=ImageAnalysis,
)
result = response.choices[0].message.parsed  # Typed object!
print(result.mood)  # Direct attribute access""", "python",
        "Native structured output with Pydantic. The .parse() method returns a typed object. No JSON parsing needed.")

    add_code_slide(prs, "Anthropic Structured Output (Tool-Use Pattern)", """response = client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=1024,
    tools=[{
        "name": "analyze_image",
        "description": "Extract structured analysis",
        "input_schema": ImageAnalysis.model_json_schema(),
    }],
    tool_choice={"type": "tool", "name": "analyze_image"},
    messages=[{"role": "user", "content": "Analyze this photo..."}],
)
data = response.content[0].input  # Dict matching schema
result = ImageAnalysis(**data)""", "python",
        "Anthropic doesn't have native structured output — this tool-use trick is the standard workaround.")

    add_code_slide(prs, "Google Gemini Structured Output", """response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Analyze this photo...",
    config={
        "response_mime_type": "application/json",
        "response_schema": ImageAnalysis,  # Pydantic model
    },
)
result = ImageAnalysis(**json.loads(response.text))""", "python",
        "Gemini uses response_mime_type + response_schema. Clean approach similar to OpenAI.")

    add_content_slide(prs, "Vision APIs — Sending Images to LLMs", [
        "All major models now support vision (image understanding)",
        "OpenAI: image_url in content array (base64 or URL)",
        "Anthropic: image source block (base64 + media_type)",
        "Google: Part.from_bytes(data=bytes, mime_type=...)",
        "Best vision: GPT-4.1, Sonnet 4.6, Gemini 2.5 Pro",
        "Use cases: OCR, analysis, accessibility, data extraction",
    ], "Vision is now standard across all providers. Show the different content formats.")

    add_code_slide(prs, "Vision — OpenAI (GPT-4.1)", """response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image}"
            }},
        ],
    }],
)""", "python",
        "OpenAI vision: mix text and image_url in the content array. Works with base64 or public URLs.")

    add_code_slide(prs, "Vision — Anthropic (Sonnet 4.6)", """response = client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64_image,
            }},
            {"type": "text", "text": "Describe this image"},
        ],
    }],
)""", "python",
        "Anthropic vision: image block with explicit media_type. Supports JPEG, PNG, GIF, WebP.")

    add_content_slide(prs, "Audio APIs", [
        "Speech-to-Text: Whisper (transcribe audio → text)",
        "Text-to-Speech: tts-1, tts-1-hd (text → spoken audio)",
        "Realtime Audio: gpt-audio-1.5 (live voice conversations)",
        "Voices: alloy, echo, fable, onyx, nova, shimmer",
        "Use case: voice assistants, transcription, accessibility",
        "Cost: Whisper ~$0.006/min, TTS ~$0.015/1K chars",
    ], "Audio APIs are mature. Whisper is excellent for transcription. TTS voices are natural-sounding.")

    add_content_slide(prs, "Image Generation — GPT Image 1.5", [
        "OpenAI's image generation via API",
        "Models: gpt-image-1, gpt-image-1.5 (latest, best quality)",
        "Sizes: 1024×1024, 1024×1792, 1792×1024",
        "Quality: standard or high (HD)",
        "Also: image editing and variations",
        "Alternative: Sora 2 for video generation",
    ], "Image gen is powerful but costs add up. Show the generate_image function from multimodal_app.py.")

    add_code_slide(prs, "Image Generation Code", """response = client.images.generate(
    model="gpt-image-1.5",
    prompt="A serene mountain landscape at sunset, photorealistic",
    size="1024x1024",
    quality="high",
    n=1,
)
image_url = response.data[0].url

# TTS Example
speech = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Hello, welcome to the AI course!",
)
speech.stream_to_file("welcome.mp3")""", "python",
        "Image gen and TTS in one slide. Both are simple API calls.")

    add_content_slide(prs, "Hands-On: Multimodal App", [
        "multimodal_app.py — analyze images with 3 providers",
        "Combines vision + structured output + TTS",
        "Run: python multimodal_app.py photo.jpg --tts",
        "Compares GPT-4.1 vs Sonnet 4.6 vs Gemini 2.5 Flash",
        "Exercise: try different images, compare structured results",
        "Bonus: python multimodal_app.py --generate 'a cat on mars'",
    ], "Hands-on exercise. Students need an image file and API keys for all 3 providers.")

    add_content_slide(prs, "Key Takeaways — Session 2", [
        "System prompts + few-shot + CoT = core prompt engineering",
        "Structured output: OpenAI native, Anthropic tool-trick, Gemini schema",
        "Vision works across all providers (slightly different formats)",
        "Audio: Whisper (STT), tts-1 (TTS), gpt-audio-1.5 (realtime)",
        "Image gen: GPT Image 1.5 via simple API call",
        "Next session: tool use & function calling",
    ], "Recap. Preview Session 3 on tool use.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-2/session-2-prompting-multimodal.pptx")
    return len(prs.slides)


# ============================================================
# SESSION 3
# ============================================================
def build_session_3():
    prs = new_prs()

    add_title_slide(prs, "Tool Use & Custom Assistants", "Function calling across providers, Assistants API, Custom GPTs", 3)

    add_content_slide(prs, "What Is Tool Use / Function Calling?", [
        "LLMs can't browse the web, check weather, or query databases",
        "Tool use: you define functions, the model decides when to call them",
        "Model outputs: function name + arguments (JSON)",
        "You execute the function and return the result",
        "Model incorporates the result into its response",
        "This is the foundation of agentic AI",
    ], "Tool use is THE most important API feature after basic chat. It's what makes AI actually useful beyond text generation.")

    add_content_slide(prs, "The Tool Use Loop (All Providers)", [
        "1. Send messages + tool definitions to the model",
        "2. Model responds with tool_calls (or text if no tool needed)",
        "3. Execute each tool call with the provided arguments",
        "4. Append tool results to the conversation",
        "5. Send back to model → it may call more tools or respond",
        "Loop continues until model gives a text response",
    ], "This loop is identical in concept across all providers. The JSON format differs slightly.")

    add_code_slide(prs, "Defining Tools (JSON Schema)", """TOOLS = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["city"],
    },
}]""", "python",
        "Tool definitions use JSON Schema. Name, description, and parameters. The description is critical — it tells the model WHEN to use the tool.")

    add_code_slide(prs, "OpenAI Tool Loop", """tools = [{"type": "function", "function": t} for t in TOOLS]

response = client.chat.completions.create(
    model="gpt-4.1", messages=messages, tools=tools)
msg = response.choices[0].message

if msg.tool_calls:
    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = execute_tool(call.function.name, args)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
    # Send back for final response...""", "python",
        "OpenAI wraps tools in {type: function, function: ...}. Tool results use role: tool with the call ID.")

    add_code_slide(prs, "Anthropic Tool Loop", """tools = [{"name": t["name"], "description": t["description"],
         "input_schema": t["parameters"]} for t in TOOLS]

response = client.messages.create(
    model="claude-sonnet-4-6-20250217", max_tokens=8192,
    tools=tools, messages=conversation)

if response.stop_reason == "tool_use":
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
    conversation.append({"role": "user", "content": tool_results})""", "python",
        "Anthropic: check stop_reason == 'tool_use'. Tool results go as a user message with tool_result blocks.")

    add_code_slide(prs, "Google Gemini Tool Loop", """from google.genai import types

tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_weather",
        description="Get current weather for a city",
        parameters=types.Schema(
            type="OBJECT",
            properties={"city": types.Schema(type="STRING")},
            required=["city"],
        )
    )
])

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=contents,
    config=types.GenerateContentConfig(tools=[tool]))

# Check for function_call in response.candidates[0].content.parts[0]""", "python",
        "Gemini uses typed objects instead of raw JSON. FunctionDeclaration + Schema. Response includes function_call.")

    add_table_slide(prs, "Tool Use — Provider Comparison",
        ["Feature", "OpenAI", "Anthropic", "Google"],
        [
            ["Tool format", "{type: function, function: ...}", "{name, input_schema}", "FunctionDeclaration"],
            ["Check for calls", "msg.tool_calls", "stop_reason == 'tool_use'", "part.function_call"],
            ["Result role", "role: 'tool'", "role: 'user' + tool_result", "FunctionResponse"],
            ["Parallel calls", "Yes (multiple)", "Yes (multiple)", "Yes"],
            ["Structured output", "Native response_format", "Tool-use trick", "response_schema"],
        ],
        "Side-by-side comparison. The concepts are identical, only the JSON structure differs.")

    add_two_column_slide(prs, "Assistants API vs Messages API",
        "OpenAI Assistants API", [
            "Server-side conversation state",
            "Built-in file search & code interpreter",
            "Threads persist across sessions",
            "Automatic tool execution loop",
            "Higher cost (storage fees)",
            "Good for: chatbots, file Q&A",
        ],
        "Messages API (All Providers)", [
            "Stateless — you manage history",
            "Full control over tool execution",
            "No storage fees",
            "Works with any provider",
            "You build the tool loop",
            "Good for: custom apps, agents",
        ],
        "Assistants API is convenient but locks you into OpenAI. Messages API gives you full control and portability.")

    add_content_slide(prs, "Custom GPTs, Skills & Gems", [
        "Custom GPTs (OpenAI): no-code wrapper — instructions + knowledge + actions",
        "Gems (Google): similar concept for Gemini",
        "Skills (OpenClaw): code-based, run tools server-side (JS/Python)",
        "Custom GPTs are easy but limited (no real code execution)",
        "Skills are powerful — full programmatic control",
        "Marketplace: GPT Store, ClawHub (covered in Session 4)",
    ], "Compare the no-code approaches. Skills (OpenClaw) are the most flexible for developers.")

    add_content_slide(prs, "Hands-On: Tool-Calling Assistant", [
        "tool_calling.py — assistant with weather, search, calculator",
        "Supports all 3 providers: --provider openai|anthropic|google",
        "Automatic tool loop — handles multi-step tool chains",
        "Try: 'What's the weather in Toronto and calculate 20% tip on $85'",
        "Watch the tool calls in the terminal output",
        "Exercise: add your own tool (e.g., unit converter)",
    ], "Hands-on. Students run tool_calling.py, try each provider, observe tool call logs.")

    add_content_slide(prs, "Key Takeaways — Session 3", [
        "Tool use = LLMs calling your functions (JSON in, JSON out)",
        "Same loop across providers: define → call → execute → return",
        "Assistants API is convenient; Messages API is flexible",
        "Custom GPTs/Skills/Gems: different packaging for the same idea",
        "Tool use is the foundation for agents (Session 5)",
        "Next session: MCP — the universal tool protocol",
    ], "Recap. Tool use is the gateway to agentic AI.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-3/session-3-tool-use.pptx")
    return len(prs.slides)


# ============================================================
# SESSION 4
# ============================================================
def build_session_4():
    prs = new_prs()

    add_title_slide(prs, "MCP, Plugins & Marketplaces", "The Model Context Protocol, MCP servers, GPT Store & ClawHub", 4)

    add_content_slide(prs, "The Problem MCP Solves", [
        "Every app builds its own tool integrations",
        "N apps × M tools = N×M custom integrations",
        "MCP: a universal protocol — build once, use everywhere",
        "Like USB for AI tools — standard plug-and-play",
        "Created by Anthropic, adopted across the ecosystem",
        "Supported in: Claude Desktop, OpenClaw, VS Code, Cursor",
    ], "MCP is the most important infrastructure development in AI tooling. Explain the N×M problem.")

    add_content_slide(prs, "How MCP Works", [
        "MCP Server: exposes tools via a standard JSON-RPC protocol",
        "MCP Client: any AI app that speaks MCP (Claude, OpenClaw, etc.)",
        "Transport: stdio (local process) or HTTP/SSE (remote)",
        "Server advertises available tools with schemas",
        "Client sends tool calls, server returns results",
        "One server can serve multiple clients simultaneously",
    ], "Architecture overview. Draw the client-server diagram on whiteboard if possible.")

    add_code_slide(prs, "MCP Server Configuration (Claude Desktop)", """{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/Users/you/Documents"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_xxx" }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres",
               "postgresql://user:pass@localhost/mydb"]
    }
  }
}""", "json",
        "This JSON config goes in Claude Desktop settings. Each server is a separate process. Show how to find the config file.")

    add_content_slide(prs, "Popular MCP Servers", [
        "Filesystem — read/write/search files on your computer",
        "GitHub — repos, issues, PRs, code search",
        "Slack — read channels, send messages, search history",
        "PostgreSQL / SQLite — query databases directly",
        "Brave Search — web search without API key setup",
        "Puppeteer — browser automation, screenshots, scraping",
    ], "These are official MCP servers from Anthropic + community. All installable via npx.")

    add_content_slide(prs, "Setting Up MCP (Live Demo)", [
        "Step 1: Install Node.js (required for npx)",
        "Step 2: Edit config file (Claude Desktop or OpenClaw)",
        "Step 3: Restart the application",
        "Step 4: Test — ask Claude to read a file or search GitHub",
        "Troubleshooting: check server logs, verify paths/tokens",
        "OpenClaw: openclaw mcp add <server-name>",
    ], "Live demo: add filesystem and GitHub MCP servers. Show the tool list appearing in the UI.")

    add_content_slide(prs, "Building Your Own MCP Server", [
        "TypeScript SDK: @modelcontextprotocol/sdk",
        "Python SDK: mcp (pip install mcp)",
        "Define tools with schemas (like function calling)",
        "Implement handlers for each tool",
        "Run as stdio process or HTTP server",
        "Publish to npm/PyPI for others to use",
    ], "Building MCP servers is straightforward. Show the basic TypeScript/Python skeleton.")

    add_code_slide(prs, "Simple MCP Server (Python)", """from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("my-tools")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="greet",
        description="Generate a greeting",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "greet":
        return [TextContent(
            type="text",
            text=f"Hello, {arguments['name']}!"
        )]

mcp.server.stdio.run(server)""", "python",
        "Minimal MCP server in Python. list_tools returns available tools, call_tool handles execution.")

    add_two_column_slide(prs, "Marketplaces — GPT Store & ClawHub",
        "GPT Store (OpenAI)", [
            "Custom GPTs published for others to use",
            "No-code: instructions + knowledge files + actions",
            "Revenue sharing for popular GPTs",
            "Discovery via search & categories",
            "Limited to OpenAI ecosystem",
        ],
        "ClawHub (OpenClaw)", [
            "Skills & MCP servers shared with community",
            "Code-based — full developer control",
            "Open ecosystem (not locked to one provider)",
            "Install with: openclaw skill install <name>",
            "Publish your own: openclaw skill publish",
        ],
        "GPT Store is consumer-friendly. ClawHub is developer-focused. Both are growing marketplaces.")

    add_content_slide(prs, "MCP vs Function Calling vs Plugins", [
        "Function calling: built into each API (provider-specific)",
        "MCP: universal protocol, works across apps and providers",
        "Plugins (deprecated): OpenAI's old approach, replaced by GPTs/Actions",
        "MCP is the future — one integration, many clients",
        "Function calling is still used inside MCP servers",
        "Best practice: build MCP servers, use them everywhere",
    ], "Clarify the relationship. MCP wraps function calling into a reusable, portable package.")

    add_content_slide(prs, "Hands-On: MCP Setup", [
        "Exercise 1: Configure filesystem + GitHub MCP servers",
        "Exercise 2: Ask Claude/OpenClaw to interact with your files",
        "Exercise 3: Browse the community MCP server list",
        "Exercise 4: (Bonus) Build a simple MCP server",
        "Reference: modelcontextprotocol.io for docs",
        "Tip: start with filesystem server — most immediately useful",
    ], "Hands-on session. Students configure MCP servers and test them.")

    add_content_slide(prs, "Key Takeaways — Session 4", [
        "MCP = universal tool protocol (USB for AI)",
        "One server → multiple clients (Claude, OpenClaw, VS Code)",
        "Config is simple JSON — command + args + env",
        "Build servers in Python or TypeScript",
        "Marketplaces: GPT Store (no-code), ClawHub (developer)",
        "Next session: agentic AI & frameworks",
    ], "Recap. MCP is infrastructure that students will use throughout the rest of the course.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-4/session-4-mcp-marketplaces.pptx")
    return len(prs.slides)


# ============================================================
# SESSION 5
# ============================================================
def build_session_5():
    prs = new_prs()

    add_title_slide(prs, "Agentic AI & Frameworks", "ReAct loop, LangGraph, CrewAI, multi-agent, and agentic coding", 5)

    add_content_slide(prs, "What Is an AI Agent?", [
        "An LLM that can reason, plan, and take actions in a loop",
        "More than chat: observe → think → act → observe → ...",
        "Uses tools to interact with the real world",
        "Can handle multi-step tasks autonomously",
        "Key difference from tool use: the model decides the workflow",
        "Examples: research agents, coding agents, data analysis agents",
    ], "Agents are the biggest shift in how we use LLMs. The model becomes a decision-maker, not just a responder.")

    add_content_slide(prs, "The ReAct Loop", [
        "ReAct = Reasoning + Acting (Yao et al., 2023)",
        "1. Thought: model reasons about what to do next",
        "2. Action: model calls a tool",
        "3. Observation: tool returns results",
        "4. Repeat until task is complete",
        "This is what every agent framework implements",
    ], "ReAct is the foundational agent pattern. Draw the loop diagram on the whiteboard.")

    add_table_slide(prs, "Agent Frameworks — February 2026",
        ["Framework", "Best For", "Language", "Key Feature"],
        [
            ["LangGraph", "Complex stateful agents", "Python/JS", "Graph-based workflow"],
            ["OpenAI Agents SDK", "Simple OpenAI agents", "Python", "Built-in tool loop"],
            ["CrewAI", "Multi-agent teams", "Python", "Role-based agents"],
            ["Anthropic Tool Use", "Claude-native agents", "Python", "Direct tool loop"],
            ["AutoGen (Microsoft)", "Conversational agents", "Python", "Agent chat"],
            ["Pydantic AI", "Type-safe agents", "Python", "Pydantic integration"],
        ],
        "Framework landscape. LangGraph is the most mature. OpenAI SDK is simplest. CrewAI is great for multi-agent.")

    add_content_slide(prs, "LangGraph — Graph-Based Agents", [
        "Model agents as directed graphs: nodes = steps, edges = flow",
        "StateGraph: define state shape, add nodes and edges",
        "Conditional edges: route based on state (e.g., tool calls → tools)",
        "Built-in persistence, streaming, and human-in-the-loop",
        "Integrates with LangChain tools and retrievers",
        "Production-ready with LangGraph Cloud",
    ], "LangGraph is the recommended framework for complex agents. Graph abstraction is powerful.")

    add_code_slide(prs, "LangGraph Agent — State & Nodes", """from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_count: int

def agent_node(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END""", "python",
        "LangGraph agent definition. State is a TypedDict. Nodes are functions. Conditional edges route the flow.")

    add_code_slide(prs, "LangGraph Agent — Building the Graph", """graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([search_web, write_report]))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["tools", END])
graph.add_edge("tools", "agent")

agent = graph.compile()

# Run it
result = agent.invoke({
    "messages": [HumanMessage("Research AI regulations")],
    "research_count": 0,
})""", "python",
        "Compile the graph and invoke it. The agent loops between reasoning and tool use until it decides to stop.")

    add_content_slide(prs, "Multi-Agent Orchestration", [
        "Multiple specialized agents collaborating on a task",
        "Patterns: supervisor (one routes to others), pipeline, debate",
        "CrewAI: define agents with roles, goals, and backstories",
        "LangGraph: use sub-graphs for each agent",
        "Example: researcher → analyst → writer → editor",
        "Challenge: coordination, context sharing, error handling",
    ], "Multi-agent is powerful but complex. Start with single agents, add more only when needed.")

    add_content_slide(prs, "Agent Memory", [
        "Short-term: conversation history (messages array)",
        "Working memory: scratchpad / state within the graph",
        "Long-term: vector DB or structured storage across sessions",
        "LangGraph: built-in checkpointing and state persistence",
        "Challenge: what to remember vs. what to forget",
        "Practical: most apps only need short-term + simple key-value",
    ], "Memory is often over-engineered. Start simple. Add long-term only when you have a real need.")

    add_two_column_slide(prs, "Agentic Coding Tools",
        "GPT-5.2-Codex (OpenAI)", [
            "Cloud-based coding agent",
            "Runs in sandboxed environment",
            "Best for: bulk refactors, test writing",
            "Async — submit task, get results later",
            "Integrated into ChatGPT & API",
        ],
        "Claude Code (Anthropic)", [
            "Terminal-based coding assistant",
            "Runs locally on your machine",
            "Best: real-time pair programming",
            "Interactive — works with you live",
            "Uses OAuth (existing subscription)",
        ],
        "Both are production-quality. Codex is async/cloud. Claude Code is interactive/local. Cursor and Copilot also popular.")

    add_content_slide(prs, "Agentic Coding — Cursor & Copilot", [
        "Cursor: AI-first code editor (VS Code fork)",
        "  - Tab completion, inline chat, multi-file edits",
        "GitHub Copilot: AI pair programmer in VS Code/JetBrains",
        "  - Copilot Chat, code suggestions, PR summaries",
        "Both use tool use under the hood (file read/write/search)",
        "Comparison deep-dive in Session 8",
    ], "Brief mention of Cursor and Copilot. Detailed comparison in Session 8.")

    add_content_slide(prs, "Hands-On: LangGraph Research Agent", [
        "langgraph_agent.py — multi-step research agent",
        "Tools: search_web, write_report",
        "Agent decides: search → analyze → search more? → write report",
        "Tracks research iterations (max 3 searches)",
        "Exercise: run it, inspect the message trace",
        "Bonus: add a new tool (e.g., save_to_file)",
    ], "Hands-on. Students run the LangGraph agent, observe the ReAct loop in action.")

    add_content_slide(prs, "Key Takeaways — Session 5", [
        "Agents = LLMs that reason + act in a loop (ReAct)",
        "LangGraph for complex agents, OpenAI SDK for simple ones",
        "Multi-agent: powerful but add complexity only when needed",
        "Memory: start with short-term, add long-term as needed",
        "Agentic coding: Codex (async), Claude Code (interactive)",
        "Next session: RAG & data retrieval",
    ], "Recap. Agents are the future but start simple.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-5/session-5-agentic-frameworks.pptx")
    return len(prs.slides)


# ============================================================
# SESSION 6
# ============================================================
def build_session_6():
    prs = new_prs()

    add_title_slide(prs, "RAG & Data", "Embeddings, vector databases, retrieval-augmented generation", 6)

    add_content_slide(prs, "Why RAG?", [
        "LLMs have training cutoffs — they don't know YOUR data",
        "RAG: retrieve relevant docs → inject into prompt → generate",
        "Gives the model accurate, up-to-date, domain-specific info",
        "No model retraining needed — just update your documents",
        "Most common production AI pattern (80%+ of enterprise apps)",
        "Alternative to fine-tuning for domain knowledge",
    ], "RAG is the most important production pattern. Most real-world AI apps use some form of RAG.")

    add_content_slide(prs, "The RAG Pipeline", [
        "1. Ingest: load documents (PDF, MD, HTML, DB records)",
        "2. Chunk: split into smaller pieces (300-500 tokens typical)",
        "3. Embed: convert chunks to vectors (text-embedding-3-small)",
        "4. Store: save vectors in a vector database",
        "5. Query: embed the question, find similar chunks",
        "6. Generate: pass retrieved chunks + question to LLM",
    ], "Walk through each step. This is the core architecture. Draw it on the whiteboard.")

    add_content_slide(prs, "Embeddings — Converting Text to Vectors", [
        "Embedding: text → dense vector (list of floats)",
        "Similar meanings → similar vectors (cosine similarity)",
        "text-embedding-3-small: 1536 dims, $0.02/1M tokens",
        "text-embedding-3-large: 3072 dims, better quality, higher cost",
        "Also: Cohere Embed, Voyage AI, open-source (BGE, E5)",
        "Key: use the SAME embedding model for documents and queries",
    ], "Embeddings are the core of semantic search. Explain cosine similarity intuitively.")

    add_code_slide(prs, "Generating Embeddings", """from openai import OpenAI
client = OpenAI()

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

# Embed a document chunk
doc_vector = embed_text("Employees get 20 days PTO per year")
# Embed a query
query_vector = embed_text("How much vacation do I get?")
# These vectors will be very similar (high cosine similarity)""", "python",
        "Simple embedding call. The key insight: semantically similar text produces similar vectors.")

    add_table_slide(prs, "Vector Database Options",
        ["Database", "Type", "Best For", "Pricing"],
        [
            ["ChromaDB", "Embedded / local", "Prototyping, small datasets", "Free (open source)"],
            ["Pinecone", "Managed cloud", "Production, zero-ops", "$0.33/hr+"],
            ["pgvector", "Postgres extension", "Existing Postgres users", "Free (ext)"],
            ["Weaviate", "Self-hosted / cloud", "Multi-modal search", "Free / managed"],
            ["Qdrant", "Self-hosted / cloud", "High performance", "Free / managed"],
        ],
        "ChromaDB for learning and prototyping. Pinecone for production. pgvector if you already use Postgres.")

    add_code_slide(prs, "ChromaDB — Store & Query", """import chromadb
chroma = chromadb.Client()  # In-memory for demo

collection = chroma.create_collection("docs",
    metadata={"hnsw:space": "cosine"})

# Store chunks with embeddings
collection.add(
    ids=["chunk_1", "chunk_2"],
    embeddings=[embed_text(chunk1), embed_text(chunk2)],
    documents=[chunk1, chunk2],
    metadatas=[{"source": "handbook.md"}, {"source": "faq.md"}],
)

# Query
results = collection.query(
    query_embeddings=[embed_text("vacation policy")],
    n_results=3,
)""", "python",
        "ChromaDB is the easiest vector DB to start with. No server needed. Show the full pipeline.")

    add_code_slide(prs, "RAG Generation — Putting It Together", """def generate_answer(query, context_chunks):
    context = "\\n\\n".join(
        f"[Source: {c['source']}]\\n{c['text']}"
        for c in context_chunks
    )
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content":
             "Answer based ONLY on the provided context. "
             "Cite sources. If unsure, say so."},
            {"role": "user", "content":
             f"Context:\\n{context}\\n\\nQuestion: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content""", "python",
        "The generation step. Key: system prompt tells the model to ONLY use provided context and cite sources.")

    add_two_column_slide(prs, "RAG vs Long Context vs Fine-Tuning",
        "When to Use RAG", [
            "Large document collections (>200K tokens total)",
            "Frequently updated data",
            "Need source citations",
            "Multiple users with different doc access",
            "Cost-sensitive (only retrieve what's needed)",
        ],
        "When NOT to Use RAG", [
            "Small corpus that fits in context window",
            "  → Just pass it all in (200K+ token windows)",
            "Need to change model behavior/style",
            "  → Fine-tuning is better",
            "Real-time data needs",
            "  → Use tool calls to live APIs",
        ],
        "RAG isn't always the answer. If your data fits in the context window, just use long context. It's simpler and often better.")

    add_content_slide(prs, "Chunking Strategies", [
        "Recursive text splitting: split on \\n\\n, then \\n, then sentences",
        "Chunk size: 300-500 tokens typical, with 50-100 token overlap",
        "Semantic chunking: split at topic boundaries (more complex)",
        "Document-aware: respect headings, code blocks, tables",
        "Too small → loses context; too large → dilutes relevance",
        "Experiment: chunk size has huge impact on retrieval quality",
    ], "Chunking is the most under-appreciated part of RAG. Bad chunking = bad retrieval = bad answers.")

    add_content_slide(prs, "Context Window Management", [
        "GPT-4.1: 1M token context, 32K output",
        "Sonnet 4.6: 200K context, 8K output (common default)",
        "Gemini 2.5 Pro: 1M+ context",
        "Strategy: use large context for small corpora, RAG for large",
        "Sliding window: keep recent + summarize old messages",
        "Token counting: tiktoken (OpenAI), anthropic token counting",
    ], "Context windows are getting huge. For many use cases, you might not need RAG at all — just stuff the context.")

    add_content_slide(prs, "Fine-Tuning (Brief Overview)", [
        "Fine-tuning: train the model on your specific data/style",
        "OpenAI: fine-tune GPT-4.1 with your JSONL training data",
        "LoRA: parameter-efficient fine-tuning for open models",
        "Best for: consistent style, specialized vocabulary, behavior",
        "NOT for: injecting facts (use RAG), general knowledge",
        "Start with prompting → RAG → fine-tune (last resort)",
    ], "Fine-tuning is usually unnecessary. RAG + good prompting covers 90% of use cases.")

    add_content_slide(prs, "Hands-On: RAG Pipeline", [
        "rag_pipeline.py — document Q&A with ChromaDB + GPT-4.1",
        "Sample docs: employee handbook, product FAQ, engineering standards",
        "Chunks → embeds → stores → retrieves → generates with citations",
        "Debug mode: see which chunks were retrieved and similarity scores",
        "Exercise: add your own documents, experiment with chunk sizes",
        "Try questions that span multiple documents",
    ], "Hands-on. Students run the RAG pipeline and experiment with their own documents.")

    add_content_slide(prs, "Key Takeaways — Session 6", [
        "RAG = retrieve → augment → generate (most common AI pattern)",
        "Embeddings: text-embedding-3-small for most use cases",
        "ChromaDB for prototyping, Pinecone/pgvector for production",
        "Chunk size matters — experiment to find the sweet spot",
        "Long context windows may eliminate need for RAG on small corpora",
        "Next session: observability, evals & security",
    ], "Recap. RAG is essential but don't over-engineer — sometimes long context is enough.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-6/session-6-rag-data.pptx")
    return len(prs.slides)


# ============================================================
# SESSION 7
# ============================================================
def build_session_7():
    prs = new_prs()

    add_title_slide(prs, "Observability, Evals & Security", "Tracing, LLM-as-judge, prompt injection, and guardrails", 7)

    add_content_slide(prs, "Why Observability Matters", [
        "LLMs are non-deterministic — same input ≠ same output",
        "You can't unit test a vibe — you need to trace and measure",
        "Without tracing: debugging is guesswork",
        "Key metrics: latency, cost, token usage, quality scores",
        "Trace every call: input, output, model, tokens, duration",
        "Traces make it possible to understand and improve your system",
    ], "Observability is often skipped but essential for production. You can't improve what you don't measure.")

    add_table_slide(prs, "Tracing & Observability Tools",
        ["Tool", "Type", "Key Feature", "Pricing"],
        [
            ["Langfuse", "Open source / cloud", "@observe decorator, cost tracking", "Free tier"],
            ["LangSmith", "Cloud (LangChain)", "Playground, datasets, evals", "Free tier"],
            ["Braintrust", "Cloud", "Evals-first, prompt mgmt", "Free tier"],
            ["Arize Phoenix", "Open source", "Traces + embeddings viz", "Free"],
        ],
        "Langfuse is recommended — open source, easy setup, works with any framework. LangSmith if you use LangChain.")

    add_content_slide(prs, "Langfuse — Decorator-Based Tracing", [
        "@observe decorator on any function → automatic tracing",
        "Drop-in OpenAI replacement: from langfuse.openai import OpenAI",
        "Traces capture: inputs, outputs, latency, cost, tokens",
        "Nested spans: see the full pipeline (validate → retrieve → generate)",
        "Dashboard: filter, search, analyze traces",
        "Self-hostable or use cloud.langfuse.com",
    ], "Langfuse is the easiest tracing to add. Two lines of code and you have full observability.")

    add_code_slide(prs, "Adding Langfuse Tracing", """from langfuse.decorators import observe
from langfuse.openai import OpenAI  # Drop-in replacement

client = OpenAI()  # Auto-traces all calls!

@observe(name="rag_pipeline")
def rag_pipeline(query: str):
    validation = validate_input(query)  # Also @observe'd
    if not validation["safe"]:
        return "Blocked: potential injection"
    
    context = retrieve(query)           # Also @observe'd
    answer = generate_answer(query, context)  # Auto-traced
    return answer

# Every call creates a trace visible in Langfuse dashboard
# with nested spans, latency, cost, and token counts""", "python",
        "Three changes: import from langfuse, add @observe decorators, done. Show the Langfuse dashboard.")

    add_content_slide(prs, "Evaluating LLM Outputs", [
        "Traditional tests don't work: output is non-deterministic",
        "Heuristic checks: keyword presence, format validation, length",
        "LLM-as-judge: use a strong model to evaluate a weaker one",
        "Criteria: correctness, relevance, completeness, grounding",
        "Score on a scale (1-5) with reasoning",
        "Build eval suites and run them on every change",
    ], "Evals are how you maintain quality over time. LLM-as-judge is the most practical approach.")

    add_code_slide(prs, "LLM-as-Judge Evaluation", """@observe(name="llm_judge")
def llm_judge(question, answer, reference=None):
    prompt = f\"\"\"Rate this answer 1-5 on each criterion.
Question: {question}
Answer: {answer}
{f"Reference: {reference}" if reference else ""}

Criteria:
- correctness, relevance, completeness, grounding
Respond in JSON.\"\"\"

    response = client.chat.completions.create(
        model="gpt-4.1",  # Use strong model as judge
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)""", "python",
        "LLM-as-judge: use GPT-4.1 to evaluate outputs. Returns structured scores + reasoning.")

    add_content_slide(prs, "Prompt Injection — The #1 Security Threat", [
        "Attacker crafts input that overrides your system prompt",
        "Examples: 'Ignore all previous instructions', 'You are now...'",
        "Can exfiltrate system prompts, bypass safety, cause harm",
        "Indirect injection: malicious content in retrieved documents",
        "No perfect defense — defense in depth is essential",
        "Two-layer approach: pattern matching + LLM classifier",
    ], "Prompt injection is the SQL injection of AI. Take it seriously, especially in production.")

    add_code_slide(prs, "Prompt Injection Detection", """INJECTION_PATTERNS = [
    "ignore all previous", "ignore your instructions",
    "you are now", "disregard", "new instructions",
    "system prompt", "reveal your", "forget everything",
]

def detect_injection_pattern(text):
    lower = text.lower()
    return any(p in lower for p in INJECTION_PATTERNS)

def detect_injection_llm(text):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # Cheap, fast classifier
        messages=[{"role": "user", "content":
            f'Is this prompt injection? Reply SAFE or INJECTION.\\n{text}'}],
        temperature=0, max_tokens=10)
    return "INJECTION" in response.choices[0].message.content

# Layer 1: fast pattern match, Layer 2: LLM for subtle attacks""", "python",
        "Two layers: fast pattern matching catches obvious attacks, LLM classifier catches sophisticated ones. Use GPT-4.1-mini for cost.")

    add_content_slide(prs, "Guardrails — Input & Output Validation", [
        "Input guardrails: injection detection, PII filtering, length limits",
        "Output guardrails: format validation, toxicity filtering, fact checking",
        "Tools: Guardrails AI, NeMo Guardrails, custom validators",
        "PII detection: regex patterns + entity recognition",
        "Toxicity: OpenAI moderation endpoint (free), Perspective API",
        "Always validate both input AND output in production",
    ], "Guardrails are your safety net. Input validation prevents attacks. Output validation prevents embarrassment.")

    add_content_slide(prs, "Building an Eval Suite", [
        "Define test cases: question, expected answer, edge cases",
        "Include: happy path, decline tests, injection tests",
        "Automate: run evals on every prompt change or code update",
        "Track scores over time — detect regressions early",
        "Gold standard: human evaluation (expensive, periodic)",
        "Practical: LLM-as-judge for daily, human for monthly review",
    ], "Eval suites are your CI/CD for AI quality. Run them automatically and track trends.")

    add_content_slide(prs, "Hands-On: Tracing + Evals", [
        "tracing_demo.py — RAG pipeline with full observability",
        "Mode 1: Interactive Q&A (traces visible in Langfuse)",
        "Mode 2: Run eval suite (correctness, decline, injection tests)",
        "Mode 3: Test injection detection (pattern + LLM classifier)",
        "Exercise: run all 3 modes, check Langfuse dashboard",
        "Bonus: add your own eval test cases",
    ], "Hands-on. Students need Langfuse credentials (free account at cloud.langfuse.com).")

    add_content_slide(prs, "Key Takeaways — Session 7", [
        "Trace everything: Langfuse @observe makes it trivial",
        "Evals: LLM-as-judge for automated quality measurement",
        "Prompt injection: two-layer defense (pattern + LLM classifier)",
        "Guardrails: validate both inputs and outputs",
        "Build eval suites and run them on every change",
        "Next session: production, cost optimization & OpenClaw",
    ], "Recap. Observability and security are non-negotiable for production AI.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-7/session-7-observability-security.pptx")
    return len(prs.slides)


# ============================================================
# SESSION 8
# ============================================================
def build_session_8():
    prs = new_prs()

    add_title_slide(prs, "Production, Dev Tools\n& OpenClaw", "Cost optimization, gateways, coding tools, and building skills", 8)

    add_content_slide(prs, "Cost Optimization — Model Routing", [
        "Not every query needs the strongest model",
        "Route by complexity: simple → GPT-4.1-nano, complex → GPT-5.2",
        "Tiered routing: nano → mini → GPT-4.1 → GPT-5.2",
        "Classifier-based: use a cheap model to decide which model to use",
        "Impact: 60-80% cost reduction with <5% quality drop",
        "Example: DeepSeek V4 at $0.14/1M for bulk processing",
    ], "Cost optimization is critical for production. Model routing is the biggest lever.")

    add_table_slide(prs, "Model Routing — Cost Impact",
        ["Query Type", "Model", "Cost/1M tokens", "Quality"],
        [
            ["Simple Q&A", "GPT-4.1-nano", "$0.10 / $0.40", "Good"],
            ["General tasks", "GPT-4.1-mini", "$0.40 / $1.60", "Very Good"],
            ["Complex analysis", "GPT-4.1", "$2.00 / $8.00", "Excellent"],
            ["Hard reasoning", "GPT-5.2", "$20 / $60", "Best"],
            ["Bulk processing", "DeepSeek V4", "$0.14 / $0.14", "Good"],
        ],
        "Show the 200x cost difference between nano and GPT-5.2. Most queries don't need the flagship model.")

    add_content_slide(prs, "Caching & Batching", [
        "Prompt caching: cache system prompts and common prefixes",
        "  - OpenAI: automatic for prompts >1024 tokens (50% discount)",
        "  - Anthropic: explicit cache_control blocks (90% discount)",
        "Semantic caching: cache responses for similar queries",
        "Batching API: OpenAI batch endpoint — 50% cost reduction",
        "  - Submit JSONL file, get results in ~24 hours",
        "Combine routing + caching + batching for maximum savings",
    ], "Three cost levers. Caching saves on repeated patterns. Batching saves on non-urgent workloads.")

    add_content_slide(prs, "AI Gateways & Routers", [
        "LiteLLM: unified API for 100+ models, load balancing, fallbacks",
        "Portkey: production gateway with caching, routing, guardrails",
        "OpenRouter: single API key for many providers",
        "Why: provider failover, cost tracking, rate limit management",
        "Pattern: your app → gateway → provider (transparent proxy)",
        "Self-host or use managed service",
    ], "Gateways abstract provider differences. Essential for production with multiple models.")

    add_content_slide(prs, "Production Concerns", [
        "Rate limits: implement exponential backoff + retry logic",
        "Streaming: essential for any user-facing application",
        "Error handling: timeouts, token limits, content filtering",
        "Logging: trace every request for debugging and compliance",
        "Privacy: where does your data go? Check provider DPAs",
        "Latency: p50 vs p99 — measure and set SLOs",
    ], "Production checklist. These are the things that bite you after demo day.")

    add_table_slide(prs, "Coding Tools Comparison — February 2026",
        ["Tool", "Type", "Model", "Best For", "Auth"],
        [
            ["Claude Code", "Terminal agent", "Sonnet 4.6 / Opus 4.6", "Pair programming", "OAuth"],
            ["GPT-5.2-Codex", "Cloud agent", "GPT-5.2-Codex", "Bulk refactors, async", "API key"],
            ["Cursor", "IDE (VS Code fork)", "Multi-model", "Editor integration", "Subscription"],
            ["GitHub Copilot", "IDE extension", "Multi-model", "Inline completions", "Subscription"],
            ["OpenClaw", "Agent platform", "Multi-model", "Skills + automation", "OAuth / API"],
        ],
        "Each tool has a different sweet spot. Claude Code for terminal work. Cursor for IDE. Codex for async bulk work.")

    add_content_slide(prs, "OpenClaw — Platform Overview", [
        "AI agent platform: chat, tools, skills, automation",
        "Supports multiple providers: OpenAI, Anthropic, Google",
        "Authentication: OAuth (existing sub) OR API key — your choice",
        "  - OAuth: openclaw setup → authorize in browser → paste code",
        "  - API key: openclaw setup → enter key → pay per token",
        "Skills: extend OpenClaw with custom tools (JS or Python)",
        "ClawHub: marketplace for sharing skills with the community",
    ], "OpenClaw overview. Emphasize the dual auth model — OAuth for convenience, API key for custom work.")

    add_content_slide(prs, "OpenClaw Skills Architecture", [
        "Skill = package of tools the AI agent can use",
        "skill.json: metadata, tool definitions (JSON Schema)",
        "index.js: tool implementations + lifecycle hooks",
        "onLoad(): initialize state, load config",
        "Tools are called by the AI automatically based on user intent",
        "Install: openclaw skill install <name> from ClawHub",
    ], "Skills are the extension mechanism. Show the skill.json + index.js structure.")

    add_code_slide(prs, "skill.json — Tool Definitions", """{
  "name": "code-snippets",
  "version": "1.0.0",
  "description": "Save & search code snippets",
  "main": "index.js",
  "tools": [{
    "name": "save_snippet",
    "description": "Save a code snippet with title and language",
    "parameters": {
      "type": "object",
      "properties": {
        "title": {"type": "string", "description": "Snippet title"},
        "code": {"type": "string", "description": "The code"},
        "language": {"type": "string", "description": "Language"}
      },
      "required": ["title", "code", "language"]
    }
  }]
}""", "json",
        "skill.json defines what the skill can do. Same JSON Schema format as function calling.")

    add_code_slide(prs, "index.js — Tool Implementation", """module.exports = {
  async onLoad(context) {
    // Initialize state, load config
    loadSnippets();
    console.log("✅ Code Snippets skill loaded");
  },

  tools: {
    async save_snippet({ title, code, language, tags = [] }) {
      const id = crypto.randomBytes(4).toString("hex");
      const snippet = { id, title, code, language, tags,
                        createdAt: new Date().toISOString() };
      snippets.push(snippet);
      saveSnippets();
      return { message: "Snippet saved!", id, title };
    },

    async search_snippets({ query, language }) {
      let results = snippets.filter(s =>
        s.title.includes(query) || s.code.includes(query));
      return { results };
    },
  },
};""", "javascript",
        "Tool implementations are async functions. Receive arguments, return results. The AI handles the UX.")

    add_content_slide(prs, "Building & Publishing Your Skill", [
        "1. Create skill directory: mkdir my-skill && cd my-skill",
        "2. Write skill.json (tool definitions) and index.js (logic)",
        "3. Test locally: openclaw skill test my-skill/",
        "4. Install locally: openclaw skill install ./my-skill/",
        "5. Publish: openclaw skill publish (to ClawHub)",
        "Idea starters: bookmark manager, API tester, note taker",
    ], "Hands-on exercise. Students build and test their own skill.")

    add_content_slide(prs, "The AI Engineer Toolkit — Full Picture", [
        "Models: GPT-5.2 / Opus 4.6 / Gemini 2.5 Pro (+ cheap alternatives)",
        "Tools: MCP servers, function calling, Skills",
        "Agents: LangGraph, OpenAI Agents SDK, CrewAI",
        "Data: RAG (embeddings + vector DB) or long context",
        "Ops: Langfuse tracing, LLM-as-judge evals, guardrails",
        "Production: model routing, caching, gateways, error handling",
    ], "The complete picture. Students now have exposure to every layer of the AI engineering stack.")

    add_content_slide(prs, "What to Build Next", [
        "Start small: a tool-calling chatbot for a specific domain",
        "Add RAG: give it access to your company's documentation",
        "Add tracing: observe how users interact, find failure modes",
        "Add evals: automated quality checks on every deployment",
        "Publish: share your MCP server or OpenClaw skill",
        "Keep learning: the field moves fast — follow changelogs!",
    ], "Practical next steps. Encourage students to build something real.")

    add_content_slide(prs, "Key Takeaways — Session 8 & Full Course", [
        "Model routing + caching + batching = massive cost savings",
        "Claude Code, Codex, Cursor — use the right tool for the job",
        "OpenClaw: OAuth or API key, skills extend functionality",
        "The AI engineer stack: models → tools → agents → data → ops",
        "Start simple, add complexity only when needed",
        "Thank you! Go build something amazing 🚀",
    ], "Final recap of the entire course. Thank the students. Q&A time.")

    prs.save("/home/lj_wsl/cut-the-crap/developer/session-8/session-8-production-openclaw.pptx")
    return len(prs.slides)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    counts = {}
    for i, builder in enumerate([
        build_session_1, build_session_2, build_session_3, build_session_4,
        build_session_5, build_session_6, build_session_7, build_session_8,
    ], 1):
        count = builder()
        counts[i] = count
        print(f"Session {i}: {count} slides")
    
    print(f"\nTotal: {sum(counts.values())} slides across 8 decks")
