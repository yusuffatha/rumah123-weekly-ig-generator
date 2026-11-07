import os
import re
import json
import csv
import datetime
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict
from openai import OpenAI

# Initialize LLM Client (Compatible with OpenAI / Gemini OpenAI-compatible endpoint)
client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
    base_url=os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
)

PROPERTY_KEYWORDS = ["kpr", "suku bunga", "ppn dtp", "tod", "lrt", "krl", "tol", "rumah123", "milenial", "gen z", "investasi", "harga rumah", "subsidi"]

def fetch_weekly_property_news(query: str = "Properti Indonesia KPR Hunian", max_items: int = 8) -> List[Dict]:
    """Fetches latest Indonesian property news via Google News RSS feed."""
    encoded_query = urllib.parse.quote(query + " when:7d")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=id&gl=ID&ceid=ID:id"
    
    req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as response:
        xml_data = response.read()
        
    root = ET.fromstring(xml_data)
    articles = []
    
    for item in root.findall('./channel/item'):
        title = item.find('title').text if item.find('title') is not None else ""
        pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
        description = item.find('description').text if item.find('description') is not None else ""
        clean_desc = re.sub(r'<[^>]+>', '', description)
        
        # Score relevance based on property keywords
        score = sum(1 for kw in PROPERTY_KEYWORDS if kw in (title + " " + clean_desc).lower())
        articles.append({
            "title": title,
            "published": pub_date,
            "summary": clean_desc,
            "relevance_score": score
        })
        
    # Sort by relevance score and return top items
    articles.sort(key=lambda x: x["relevance_score"], reverse=True)
    return articles[:max_items]

def extract_verified_facts(articles: List[Dict]) -> Dict:
    """Stage 1 LLM: Strict low-temperature fact extraction from raw news."""
    news_context = "\n".join([f"- [{a['published']}] {a['title']} | {a['summary']}" for a in articles])
    
    prompt = f"""You are a Senior Property Market Data Analyst at 99 Group (Rumah123).
Analyze the following Indonesian property news headlines from the past 7 days:
{news_context}

Task:
1. Identify the top 2 most impactful trends or policy updates relevant to Indonesian property seekers (e.g., KPR rates, government incentives, infrastructure/TOD, market shifts).
2. Extract ONLY verifiable facts stated in the headlines. Do NOT invent interest rates or statistics.
3. Return output strictly in valid JSON with keys: "dominant_theme", "key_facts" (list of strings), "target_segment", and "why_it_matters"."""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_text = response.choices[0].message.content
    clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(clean_json)

def generate_rumah123_ig_post(facts_data: Dict) -> Dict:
    """Stage 2 LLM: Brand-aligned Instagram Carousel and Caption Generator."""
    prompt = f"""You are the Lead Social Media Copywriter for Rumah123 (Indonesia's #1 Property Platform).
Using ONLY the verified weekly property facts below, create an engaging 5-slide Instagram Carousel and Caption.

Verified Weekly Facts:
{json.dumps(facts_data, indent=2, ensure_ascii=False)}

Brand Guidelines:
- Tone: Professional, trustworthy, and data-backed, yet warm, witty, relatable, and conversational in Bahasa Indonesia.
- Structure:
  * Slide 1: Scroll-stopping Hook question/statement about this week's news.
  * Slide 2: Trend/News Insight #1 explained simply for homebuyers.
  * Slide 3: Trend/News Insight #2 & financial/location impact.
  * Slide 4: Actionable tips using Rumah123 features (e.g., Simulasi KPR, Filter Lokasi, Listing Terverifikasi).
  * Slide 5: Interactive Poll / Question CTA to drive comments.
- Output strictly as valid JSON with keys:
  "week_id", "target_audience", "carousel_slides" (array of objects with "slide_number", "headline", "subtext", "visual_prompt"), and "instagram_caption"."""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        temperature=0.6,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_text = response.choices[0].message.content
    clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(clean_json)

def export_outputs(post_data: Dict):
    """Exports JSON payload and Canva Bulk Create CSV."""
    os.makedirs("output", exist_ok=True)
    with open("output/weekly_ig_post.json", "w", encoding="utf-8") as f:
        json.dump(post_data, f, indent=2, ensure_ascii=False)
        
    with open("output/canva_bulk_slides.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_number", "headline", "subtext", "visual_prompt"])
        for slide in post_data.get("carousel_slides", []):
            writer.writerow([slide["slide_number"], slide["headline"], slide["subtext"], slide["visual_prompt"]])

if __name__ == "__main__":
    news = fetch_weekly_property_news()
    facts = extract_verified_facts(news)
    ig_post = generate_rumah123_ig_post(facts)
    export_outputs(ig_post)
    print(json.dumps(ig_post, indent=2, ensure_ascii=False))
