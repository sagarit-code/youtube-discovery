from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, List, Dict, Any
from googleapiclient.discovery import build
from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# ----------------- CONFIG -----------------
llm =ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

youtube = build(
    "youtube",
    "v3",
    developerKey=os.getenv("YOUTUBE_API_KEY")
)

# ----------------- STATE -----------------
class AgentState(TypedDict):
    query: str
    intent: Dict[str, Any]
    video_ids: List[str]
    videos: List[Dict]
    channels: List[Dict]
    final_results: List[Dict]

# ----------------- NODE 1: INTENT PARSER -----------------
import json

def parse_intent(state: AgentState) -> AgentState:
    prompt = f"""
Extract structured intent from this query:

"{state['query']}"

Return JSON only with:
- niche
- country
- format (shorts/long)
- max_subscribers
- metric_priority
"""

    response = llm.invoke([
        SystemMessage(content="You are a strict JSON generator. No markdown."),
        HumanMessage(content=prompt)
    ])

    state["intent"] = json.loads(response.content)
    return state


# ----------------- NODE 2: DISCOVERY -----------------
def discover_videos(state: AgentState) -> AgentState:
    published_after = (datetime.utcnow() - timedelta(days=30)).isoformat("T") + "Z"

    results = youtube.search().list(
        part="snippet",
        q=state["intent"]["niche"],
        type="video",
        videoDuration="short" if state["intent"]["format"] == "shorts" else "any",
        regionCode="IN",
        publishedAfter=published_after,
        maxResults=15
    ).execute()

    video_ids = list({
        item["id"]["videoId"]
        for item in results["items"]
    })

    state["video_ids"] = video_ids
    return state

# ----------------- NODE 3: VIDEO ENRICHMENT -----------------
def fetch_video_stats(state: AgentState) -> AgentState:
    response = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(state["video_ids"])
    ).execute()

    state["videos"] = response["items"]
    return state

# ----------------- NODE 4: CHANNEL ENRICHMENT -----------------
def fetch_channels(state: AgentState) -> AgentState:
    channel_ids = list({
        v["snippet"]["channelId"]
        for v in state["videos"]
    })

    response = youtube.channels().list(
        part="snippet,statistics",
        id=",".join(channel_ids)
    ).execute()

    state["channels"] = response["items"]
    return state

# ----------------- NODE 5: AI EVALUATOR -----------------
def evaluate(state: AgentState) -> AgentState:
    prompt = f"""
Analyze creators.

CHANNELS:
{state['channels']}

Rules:
- subscribers < {state['intent']['max_subscribers']}
- strong engagement

Return JSON list with:
- channel_name
- subscribers
- avg_views
- reasoning
"""

    response = llm.invoke([
        SystemMessage(content="Return only valid JSON."),
        HumanMessage(content=prompt)
    ])

    state["final_results"] = json.loads(response.content)
    return state


# ----------------- GRAPH -----------------
graph = StateGraph(AgentState)

graph.add_node("intent", parse_intent)
graph.add_node("discover", discover_videos)
graph.add_node("videos", fetch_video_stats)
graph.add_node("channels", fetch_channels)
graph.add_node("evaluate", evaluate)

graph.set_entry_point("intent")
graph.add_edge("intent", "discover")
graph.add_edge("discover", "videos")
graph.add_edge("videos", "channels")
graph.add_edge("channels", "evaluate")
graph.add_edge("evaluate", END)

agent = graph.compile()

# ----------------- RUN -----------------
if __name__ == "__main__":
    result = agent.invoke({
        "query": "Find Indian fashion creators under 500k subs whose shorts are growing fast"
    })

    for r in result["final_results"]:
        print("\n🔥", r["channel_name"])
        print("Subs:", r["subscribers"])
        print("Avg Views:", r["avg_views"])
        print("Why:", r["reasoning"])
