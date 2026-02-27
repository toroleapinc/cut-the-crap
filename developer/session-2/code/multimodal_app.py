"""
Cut the Crap — Session 2: Multimodal Analysis App
Sends an image to OpenAI, Anthropic, and Google, compares results,
extracts structured data, and optionally generates TTS narration.

Updated: February 2026
Models: GPT-4.1, Claude Sonnet 4.6, Gemini 2.5 Flash

Requirements:
    pip install openai anthropic google-genai pydantic
    export OPENAI_API_KEY=your-key
    export ANTHROPIC_API_KEY=your-key
    export GOOGLE_API_KEY=your-key
"""

import base64
import json
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum


# --- Structured Output Schema ---

class Mood(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    DRAMATIC = "dramatic"
    PEACEFUL = "peaceful"
    ENERGETIC = "energetic"

class DetectedObject(BaseModel):
    name: str = Field(description="Object name")
    confidence: str = Field(description="high, medium, or low")

class ImageAnalysis(BaseModel):
    """Structured analysis of an image."""
    description: str = Field(description="One paragraph description of the image")
    objects: list[DetectedObject] = Field(description="Objects detected in the image")
    dominant_colors: list[str] = Field(max_length=5, description="Top dominant colors")
    mood: Mood
    scene_type: str = Field(description="e.g., indoor, outdoor, portrait, landscape")
    text_visible: bool = Field(description="Whether any text is visible in the image")


def encode_image(path: str) -> str:
    """Read and base64 encode an image file."""
    return base64.b64encode(Path(path).read_bytes()).decode()


# --- Provider Analyzers ---

def analyze_openai(image_path: str) -> tuple[str, ImageAnalysis | None]:
    """Analyze image with GPT-4.1. Returns (description, structured_data).
    
    GPT-4.1 supports vision natively — just pass image_url in content array.
    Structured Outputs use response_format with a Pydantic model.
    """
    from openai import OpenAI
    client = OpenAI()
    b64 = encode_image(image_path)

    # Step 1: Get text description
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail in one paragraph."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }},
            ],
        }],
    )
    description = response.choices[0].message.content

    # Step 2: Get structured output (native structured outputs — GA since GPT-4o)
    structured_response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and extract structured data."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }},
            ],
        }],
        response_format=ImageAnalysis,
    )
    structured = structured_response.choices[0].message.parsed

    return description, structured


def analyze_anthropic(image_path: str) -> tuple[str, ImageAnalysis | None]:
    """Analyze image with Claude Sonnet 4.6.
    
    Anthropic vision: pass base64 image in content blocks.
    Structured output: use tool_choice to force a tool call with your schema.
    """
    import anthropic
    client = anthropic.Anthropic()
    b64 = encode_image(image_path)

    # Determine media type
    suffix = Path(image_path).suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "image/jpeg")

    # Step 1: Text description
    response = client.messages.create(
        model="claude-sonnet-4-6-20250217",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": b64,
                }},
                {"type": "text", "text": "Describe this image in detail in one paragraph."},
            ],
        }],
    )
    description = response.content[0].text

    # Step 2: Structured output via tool use (Anthropic's pattern for structured output)
    structured_response = client.messages.create(
        model="claude-sonnet-4-6-20250217",
        max_tokens=1024,
        tools=[{
            "name": "analyze_image",
            "description": "Extract structured analysis from an image",
            "input_schema": ImageAnalysis.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "analyze_image"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": b64,
                }},
                {"type": "text", "text": "Analyze this image and extract structured data."},
            ],
        }],
    )
    data = structured_response.content[0].input
    structured = ImageAnalysis(**data)

    return description, structured


def analyze_google(image_path: str) -> tuple[str, ImageAnalysis | None]:
    """Analyze image with Gemini 2.5 Flash.
    
    Google GenAI SDK: pass image bytes via types.Part.from_bytes().
    Structured output: use response_mime_type + response_schema.
    """
    from google import genai
    from google.genai import types
    client = genai.Client()
    image_bytes = Path(image_path).read_bytes()

    suffix = Path(image_path).suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    mime_type = mime.get(suffix.lstrip("."), "image/jpeg")

    # Step 1: Text description
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Describe this image in detail in one paragraph.",
        ],
    )
    description = response.text

    # Step 2: Structured output
    structured_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Analyze this image and extract structured data.",
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": ImageAnalysis,
        },
    )
    structured = ImageAnalysis(**json.loads(structured_response.text))

    return description, structured


# --- TTS Narration ---

def generate_narration(text: str, output_path: str = "narration.mp3"):
    """Generate TTS narration of the image description using OpenAI.
    
    OpenAI TTS models (Feb 2026):
      - tts-1      : Standard quality, low latency
      - tts-1-hd   : Higher quality
      - gpt-audio-1.5 : Latest audio model (Realtime API)
    """
    from openai import OpenAI
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text[:4096],  # TTS has input length limits
    )
    response.stream_to_file(output_path)
    print(f"🔊 Narration saved to {output_path}")


# --- Image Generation ---

def generate_image(prompt: str, output_path: str = "generated.png"):
    """Generate an image from a text prompt using OpenAI GPT Image 1.
    
    Models (Feb 2026):
      - gpt-image-1   : Standard image generation
      - gpt-image-1.5 : Latest, highest quality
    """
    from openai import OpenAI
    client = OpenAI()
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="high",
        n=1,
    )
    # Save image
    import urllib.request
    urllib.request.urlretrieve(response.data[0].url, output_path)
    print(f"🎨 Image saved to {output_path}")


# --- Main ---

def print_analysis(provider: str, description: str, structured: ImageAnalysis | None):
    print(f"\n{'='*60}")
    print(f"  {provider}")
    print(f"{'='*60}")
    print(f"\n📝 Description:\n{description}")
    if structured:
        print(f"\n📊 Structured Analysis:")
        print(f"  Scene: {structured.scene_type}")
        print(f"  Mood: {structured.mood.value}")
        print(f"  Colors: {', '.join(structured.dominant_colors)}")
        print(f"  Objects: {', '.join(o.name for o in structured.objects)}")
        print(f"  Text visible: {structured.text_visible}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python multimodal_app.py <image_path> [--tts] [--generate <prompt>]")
        print("Example: python multimodal_app.py photo.jpg --tts")
        print("Example: python multimodal_app.py --generate 'a cat on the moon'")
        sys.exit(1)

    # Handle image generation mode
    if sys.argv[1] == "--generate":
        prompt = " ".join(sys.argv[2:])
        generate_image(prompt)
        return

    image_path = sys.argv[1]
    do_tts = "--tts" in sys.argv

    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    print(f"🖼️  Analyzing: {image_path}")
    print(f"   Sending to OpenAI, Anthropic, and Google...\n")

    results = {}

    # Analyze with each provider (try each, don't fail on one)
    for name, fn in [("OpenAI (GPT-4.1)", analyze_openai),
                     ("Anthropic (Claude Sonnet 4.6)", analyze_anthropic),
                     ("Google (Gemini 2.5 Flash)", analyze_google)]:
        try:
            desc, structured = fn(image_path)
            results[name] = (desc, structured)
            print_analysis(name, desc, structured)
        except Exception as e:
            print(f"\n⚠️  {name} failed: {e}")

    # TTS narration of the first successful result
    if do_tts and results:
        first_desc = next(iter(results.values()))[0]
        print(f"\n🎙️  Generating narration...")
        generate_narration(first_desc)

    print(f"\n{'='*60}")
    print(f"  Done! Analyzed with {len(results)}/3 providers.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
