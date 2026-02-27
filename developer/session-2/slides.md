# Session 2: Prompting, Structured Output & Multimodal
## Cut the Crap — AI Engineer Edition

---

## Slide 1: Session 2 Overview

**SHOW:**
```
Today:
1. Prompt engineering that actually works
2. Getting reliable JSON from LLMs
3. Multimodal: vision, audio, image generation
4. Hands-on: build a multimodal app
```

**SAY:**
> Last session you made API calls. Today we level up. Three big topics: prompt engineering — not the "10 magic prompts" clickbait, but real techniques that work in production. Structured output — how to get reliable JSON back, every time. And multimodal — sending images, generating audio, the whole stack.

---

## Slide 2: System Prompts (Topic 8)

**SHOW:**
```python
# OpenAI — system prompt in messages array
messages = [
    {"role": "system", "content": "You are a senior Python developer. "
     "Answer only with code. No explanations unless asked."},
    {"role": "user", "content": "Parse this CSV and find duplicates"}
]

# Anthropic — system as separate parameter
response = client.messages.create(
    model="claude-sonnet-4-6-20250217",
    system="You are a senior Python developer. "
           "Answer only with code. No explanations unless asked.",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Parse this CSV and find duplicates"}]
)

# Google — system_instruction parameter
response = client.models.generate_content(
    model="gemini-3-pro",
    config={"system_instruction": "You are a senior Python developer. "
            "Answer only with code. No explanations unless asked."},
    contents="Parse this CSV and find duplicates"
)
```

**SAY:**
> The system prompt is your most powerful tool. It sets the model's persona, constraints, and output format. Three providers, three places to put it. The actual content is the same — it's just the API shape that differs. A good system prompt is specific: don't say "be helpful," say "You are a senior Python developer. Answer only with code."

---

## Slide 3: Prompt Engineering Techniques

**SHOW:**
```
1. BE SPECIFIC
   ❌ "Summarize this"
   ✅ "Summarize in 3 bullet points, max 20 words each, focus on action items"

2. FEW-SHOT EXAMPLES
   "Classify the sentiment:
    'Great product!' → positive
    'Worst experience ever' → negative
    'It was okay I guess' → neutral
    'The delivery was late but the food was amazing' → ???"

3. CHAIN-OF-THOUGHT
   "Think step by step before answering."
   "First, identify the key variables. Then, set up the equation..."

4. OUTPUT FORMAT
   "Respond in this exact JSON format: {\"summary\": ..., \"score\": ...}"
   
5. NEGATIVE CONSTRAINTS
   "Do NOT include disclaimers. Do NOT say 'As an AI...'"
```

**SAY:**
> Five techniques that cover 90% of what you need. Being specific is obvious but most people don't do it. Few-shot examples are incredibly powerful — show the model what you want with 2-3 examples. Chain-of-thought — just saying "think step by step" measurably improves accuracy on reasoning tasks. Specifying output format prevents the model from wrapping your JSON in markdown. And negative constraints — telling the model what NOT to do — are often more effective than positive instructions.

---

## Slide 4: Advanced Prompting Patterns

**SHOW:**
```python
# Role stacking — multiple perspectives
system = """You are three experts debating:
1. A security engineer focused on risks
2. A product manager focused on user experience  
3. A performance engineer focused on speed

For each question, give all three perspectives, then a final recommendation."""

# Structured reasoning template
system = """For each question:
<analysis>
[Your step-by-step reasoning here — the user won't see this]
</analysis>

<answer>
[Your final, concise answer]
</answer>"""

# XML tags work great with Claude, good with others too
user_prompt = """Analyze this code for bugs:
<code>
{code_here}
</code>

<context>
This runs in production handling 10K requests/second.
</context>"""
```

**SAY:**
> A few advanced patterns. Role stacking gives you multiple perspectives from one call. The structured reasoning template — especially with XML tags — lets the model think but gives you a clean output to parse. XML tags are particularly effective with Claude but work well across all models. Use them to clearly delimit sections of your input.

---

## Slide 5: Structured Output — The Problem (Topic 9)

**SHOW:**
```python
# What you want:
{"name": "Alice", "age": 30, "skills": ["python", "sql"]}

# What you often get:
"Here's the JSON you requested:\n```json\n{\"name\": \"Alice\"..."

# Or worse:
"Sure! Here is the information:\n- Name: Alice\n- Age: 30\n..."

# Or WORST:
{"name": "Alice", "age": "thirty", "skills": "python and sql"}
# (wrong types!)
```

```
The problem: LLMs output TEXT. You need DATA.
Solutions (from worst to best):
1. "Please return JSON" in prompt → unreliable
2. JSON mode → valid JSON, but no schema enforcement
3. Structured output / response_format → guaranteed schema
4. Tool use with strict schema → most reliable
```

**SAY:**
> Here's the reality: you ask for JSON, you get JSON wrapped in markdown. Or you get JSON with wrong types. Or the model decides to be "helpful" and gives you prose instead. This is a solved problem in 2026, but you need to use the right tools.

---

## Slide 6: OpenAI Structured Output

**SHOW:**
```python
from pydantic import BaseModel
from openai import OpenAI

class UserProfile(BaseModel):
    name: str
    age: int
    skills: list[str]
    experience_years: float
    is_employed: bool

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": "Extract user profile from the text."},
        {"role": "user", "content": "Alice is 30, knows Python and SQL, "
         "has been coding for 8.5 years, currently working at Acme Corp."}
    ],
    response_format=UserProfile,  # ← Pydantic model!
)

profile = response.choices[0].message.parsed  # ← Already a Python object!
print(profile.name)        # "Alice"
print(profile.age)         # 30
print(profile.skills)      # ["Python", "SQL"]
print(profile.is_employed)  # True
```

**SAY:**
> OpenAI's structured output is the cleanest. Define a Pydantic model — that's your schema. Pass it as `response_format`. The response comes back already parsed into your Python object. No JSON parsing, no validation — it's guaranteed to match your schema. This uses constrained decoding under the hood — the model literally cannot produce invalid output.

---

## Slide 7: Anthropic Structured Output

**SHOW:**
```python
import anthropic
import json
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    skills: list[str]
    experience_years: float
    is_employed: bool

client = anthropic.Anthropic()

# Approach 1: Tool use trick (most reliable)
response = client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=1024,
    tools=[{
        "name": "extract_profile",
        "description": "Extract a user profile from text",
        "input_schema": UserProfile.model_json_schema(),
    }],
    tool_choice={"type": "tool", "name": "extract_profile"},
    messages=[{"role": "user", "content": "Alice is 30, knows Python and SQL, "
               "has been coding for 8.5 years, currently at Acme Corp."}]
)

# The model is forced to call the tool with valid schema
data = response.content[0].input  # dict matching schema
profile = UserProfile(**data)
print(profile.name)  # "Alice"
```

**SAY:**
> Anthropic doesn't have a `response_format` parameter like OpenAI. The reliable pattern is to use tool use as a structured output mechanism. Define a tool with your Pydantic schema, force the model to call it with `tool_choice`, and you get validated data back. It's a clever trick — you're not actually calling a function, you're just using the tool schema to constrain the output.

---

## Slide 8: Google Structured Output

**SHOW:**
```python
from google import genai
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    skills: list[str]
    experience_years: float
    is_employed: bool

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-pro",
    contents="Alice is 30, knows Python and SQL, "
             "has been coding for 8.5 years, currently at Acme Corp.",
    config={
        "response_mime_type": "application/json",
        "response_schema": UserProfile,
    }
)

import json
profile = UserProfile(**json.loads(response.text))
print(profile.name)  # "Alice"
```

**SAY:**
> Google uses `response_mime_type` set to JSON plus a `response_schema`. You can pass a Pydantic model directly. All three providers now support this, they just spell it differently. The key insight: always use Pydantic models to define your schemas. They work everywhere, they validate data, and they give you type hints in your IDE.

---

## Slide 9: Pydantic Schema Patterns

**SHOW:**
```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Entity(BaseModel):
    name: str = Field(description="The entity name")
    type: str = Field(description="PERSON, ORG, or LOCATION")
    confidence: float = Field(ge=0, le=1, description="0-1 confidence score")

class AnalysisResult(BaseModel):
    """Structured analysis of a text document."""
    summary: str = Field(description="One-sentence summary")
    sentiment: Sentiment
    entities: list[Entity]
    key_topics: list[str] = Field(max_length=5, description="Top 5 topics")
    language: str = Field(description="ISO 639-1 language code")
    word_count: int
    contains_pii: bool
    pii_types: Optional[list[str]] = Field(
        default=None,
        description="Types of PII found, if any"
    )
```

**SAY:**
> This is what production schemas look like. Use Enums to constrain values to a fixed set. Use Field descriptions — they actually help the model understand what you want. Use Optional for fields that might not apply. Nested models handle complex structures. The `ge` and `le` on confidence constrain the range. This schema works with all three providers.

---

## Slide 10: Multimodal — Vision (Topic 10)

**SHOW:**
```python
# === OpenAI Vision ===
import base64

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image('photo.jpg')}"
            }}
        ]
    }]
)

# === Anthropic Vision ===
response = anthropic_client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encode_image("photo.jpg"),
            }},
            {"type": "text", "text": "What's in this image?"},
        ]
    }]
)

# === Google Vision (simplest) ===
from google.genai import types
from pathlib import Path

response = google_client.models.generate_content(
    model="gemini-3-pro",
    contents=[
        types.Part.from_bytes(data=Path("photo.jpg").read_bytes(), mime_type="image/jpeg"),
        "What's in this image?"
    ]
)
```

**SAY:**
> All three major providers support vision. Send an image with your text, and the model "sees" it. OpenAI and Anthropic use base64 encoding — slightly different message formats. Google's is the simplest. Vision is incredibly useful: analyze screenshots, extract data from documents, read charts, moderate content. The quality is excellent across all three.

---

## Slide 11: Audio — Whisper & TTS

**SHOW:**
```python
# === Speech-to-Text (Whisper) ===
from openai import OpenAI
client = OpenAI()

with open("meeting.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",  # includes timestamps
    )
print(transcript.text)

# === Text-to-Speech ===
response = client.audio.speech.create(
    model="tts-1",
    voice="nova",        # alloy, echo, fable, onyx, nova, shimmer
    input="Hello! This is synthesized speech.",
    response_format="mp3",
)
response.stream_to_file("output.mp3")

# === Google Audio (Gemini native) ===
# Gemini 3 Pro can process audio natively
response = google_client.models.generate_content(
    model="gemini-3-pro",
    contents=[
        types.Part.from_bytes(data=Path("audio.mp3").read_bytes(), mime_type="audio/mp3"),
        "Transcribe this audio and summarize the key points."
    ]
)
```

**SAY:**
> Audio is a two-way street. Whisper transcribes speech to text — it's shockingly good, handles accents, multiple languages, even noisy environments. TTS goes the other direction — text to speech with multiple voices. Google's Gemini can handle audio natively as input, just like images. The combination is powerful: transcribe a meeting, summarize it, generate action items — all API calls.

---

## Slide 12: Image Generation

**SHOW:**
```python
# === DALL-E 3 (OpenAI) ===
response = client.images.generate(
    model="dall-e-3",
    prompt="A developer at their desk with multiple monitors showing code, "
           "digital art style, warm lighting",
    size="1024x1024",
    quality="hd",
    n=1,
)
image_url = response.data[0].url

# === Google Imagen 3 ===
response = google_client.models.generate_images(
    model="imagen-3.0-generate-002",
    prompt="A developer at their desk with multiple monitors showing code",
    config={"number_of_images": 1},
)
# response.generated_images[0].image.image_bytes

# Anthropic: No image generation API (by design — focused on text/analysis)
```

```
Image Generation Tips:
- Be descriptive: style, lighting, composition, mood
- Specify what you DON'T want: "no text, no watermarks"
- DALL-E 3 rewrites your prompt (can see revised_prompt in response)
- Imagen 3 excels at photorealism
- Neither is great at text-in-images (improving though)
```

**SAY:**
> Image generation is OpenAI and Google territory — Anthropic deliberately doesn't do it. DALL-E 3 is the most well-known. Imagen 3 from Google is arguably better at photorealism. A key thing: DALL-E 3 rewrites your prompt to be more detailed — you can see what it actually used in `revised_prompt`. For production use, always moderate generated images.

---

## Slide 13: Hands-On — Multimodal App (Topic 11)

**SHOW:**
```
📝 Exercise: Build a multimodal analysis app

The app should:
1. Accept an image (file path or URL)
2. Analyze it with GPT-5.2, Claude, AND Gemini
3. Compare the three descriptions side by side
4. Extract structured data (objects, colors, mood) using Pydantic
5. Optionally generate a TTS narration of the description

Time: 25 minutes
Starter code: session-2/code/multimodal_app.py
```

**SAY:**
> Your exercise: build a multimodal app that sends the same image to all three providers and compares their analyses. Then extract structured data from the descriptions. This combines everything from today — multimodal input, structured output, and comparing providers. Open the starter code and let's go.

---

## Slide 14: Session 2 Recap

**SHOW:**
```
✅ System prompts: your most powerful tool — be specific
✅ Few-shot > zero-shot for consistent output
✅ Chain-of-thought: "think step by step" works
✅ Structured output: use Pydantic + provider-specific features
   - OpenAI: response_format= (cleanest)
   - Anthropic: tool use trick
   - Google: response_mime_type + response_schema
✅ Vision: all 3 providers, slightly different message formats
✅ Audio: Whisper (STT), TTS, Gemini native audio
✅ Image gen: DALL-E 3, Imagen 3

Next session: Tool use, function calling, custom assistants
```

**SAY:**
> Session 2 done. You now know how to engineer prompts properly, get reliable structured data from any provider, and work with images and audio. The Pydantic trick for Anthropic structured output is worth remembering — it's the cleanest cross-provider pattern. Next session we get into tool use and function calling — that's where AI goes from "text generator" to "actually does things."
