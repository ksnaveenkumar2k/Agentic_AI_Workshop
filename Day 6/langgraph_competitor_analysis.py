import streamlit as st
from typing import TypedDict, List, Dict, Any, Optional
import requests
import random
from datetime import datetime, timedelta
import math
import logging
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import numpy as np
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Enhanced State Dict for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[str], add_messages]
    competitor_data: Dict[str, Any]
    market_data: Dict[str, Any]
    investor_analysis: Dict[str, Any]
    location: str
    query: str
    iteration_count: int
    agent_outputs: List[Dict[str, Any]]


@dataclass
class MarketMetrics:
    market_size: float
    growth_rate: float
    competition_density: int
    avg_revenue_per_store: float
    market_saturation: float
    opportunity_score: float


@dataclass
class InvestmentAnalysis:
    roi_projection: float
    payback_period: float
    risk_score: float
    investment_recommendation: str
    confidence_level: float
    key_factors: List[str]


# Utility functions (unchanged)
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


CATEGORY_MAP = {
    "clothing": {"tag": "shop=clothes", "avg_revenue": 250000, "startup_cost": 150000},
    "apparel": {"tag": "shop=clothes", "avg_revenue": 250000, "startup_cost": 150000},
    "fashion": {"tag": "shop=clothes", "avg_revenue": 300000, "startup_cost": 200000},
    "sportswear": {"tag": "shop=sports", "avg_revenue": 180000, "startup_cost": 120000},
    "shoes": {"tag": "shop=shoes", "avg_revenue": 200000, "startup_cost": 100000},
    "restaurant": {
        "tag": "amenity=restaurant",
        "avg_revenue": 400000,
        "startup_cost": 250000,
    },
    "cafe": {"tag": "amenity=cafe", "avg_revenue": 150000, "startup_cost": 80000},
    "grocery": {
        "tag": "shop=supermarket",
        "avg_revenue": 500000,
        "startup_cost": 300000,
    },
    "pharmacy": {
        "tag": "amenity=pharmacy",
        "avg_revenue": 350000,
        "startup_cost": 200000,
    },
}


def map_query_to_category_data(query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    for key, val in CATEGORY_MAP.items():
        if key in query_lower:
            return val
    return {"tag": "shop=clothes", "avg_revenue": 200000, "startup_cost": 150000}


def geocode_location(location: str) -> Dict[str, float]:
    logger.debug(f"Geocoding location: {location}")
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    resp = requests.get(
        url, params=params, headers={"User-Agent": "investor-market-analysis-tool"}
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"Location '{location}' not found.")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    logger.debug(f"Geocoded {location} to lat: {lat}, lon: {lon}")
    return {"lat": lat, "lon": lon}


def query_overpass(
    lat: float, lon: float, category_tag: str, radius: int = 5000
) -> List[Dict[str, Any]]:
    logger.debug(
        f"Querying Overpass for category: {category_tag} around lat:{lat}, lon:{lon}, radius:{radius}m"
    )
    key, value = category_tag.split("=")
    query = f"""
    [out:json][timeout:25];
    (
      node[{key}={value}](around:{radius},{lat},{lon});
      way[{key}={value}](around:{radius},{lat},{lon});
      relation[{key}={value}](around:{radius},{lat},{lon});
    );
    out center;
    """
    url = "https://overpass-api.de/api/interpreter"
    resp = requests.post(
        url,
        data=query.encode("utf-8"),
        headers={"User-Agent": "investor-market-analysis-tool"},
    )
    resp.raise_for_status()
    result = resp.json()
    elements = result.get("elements", [])
    logger.debug(f"Found {len(elements)} elements from Overpass")
    return elements


def analyze_competitors(
    elements: List[Dict], lat: float, lon: float, category_data: Dict
) -> List[Dict[str, Any]]:
    competitors = []
    for el in elements:
        props = el.get("tags", {})
        name = props.get("name", "Unknown Business")
        addr_parts = []
        for k in ["addr:housenumber", "addr:street", "addr:city"]:
            if k in props:
                addr_parts.append(props[k])
        address = ", ".join(addr_parts) if addr_parts else "Address not available"

        if el.get("lat") and el.get("lon"):
            plat, plon = el["lat"], el["lon"]
        elif "center" in el:
            plat, plon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue

        distance_km = haversine(lat, lon, plat, plon)
        base_footfall = random.randint(300, 1500)
        competitor = {
            "name": name,
            "address": address,
            "lat": plat,
            "lon": plon,
            "distance_km": round(distance_km, 2),
            "footfall_daily_avg": base_footfall,
            "peak_hours": simulate_peak_hours(category_data["tag"]),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "estimated_monthly_revenue": base_footfall * random.randint(15, 45) * 30,
            "market_share_estimate": round(random.uniform(2, 15), 2),
            "competitive_strength": calculate_competitive_strength(
                base_footfall, distance_km
            ),
            "category": category_data["tag"],
        }
        competitors.append(competitor)
    return sorted(competitors, key=lambda x: x["distance_km"])


def calculate_competitive_strength(footfall: int, distance: float) -> str:
    score = (footfall / 100) - (distance * 2)
    if score > 10:
        return "High"
    elif score > 5:
        return "Medium"
    else:
        return "Low"


def simulate_peak_hours(category: str) -> List[str]:
    if "restaurant" in category or "cafe" in category:
        return random.choice(
            [
                ["12:00-14:00", "19:00-21:00"],
                ["11:30-13:30", "18:00-20:30"],
                ["13:00-15:00", "20:00-22:00"],
            ]
        )
    else:
        return random.choice(
            [
                ["10:00-13:00", "16:00-19:00"],
                ["11:00-14:00", "17:00-20:00"],
                ["12:00-15:00", "18:00-21:00"],
            ]
        )


def analyze_market_metrics(
    competitors: List[Dict], category_data: Dict, location: str
) -> MarketMetrics:
    total_competitors = len(competitors)
    avg_footfall = (
        np.mean([c["footfall_daily_avg"] for c in competitors]) if competitors else 0
    )
    market_size = total_competitors * category_data["avg_revenue"]
    growth_rate = random.uniform(3.5, 12.5)
    competition_density = total_competitors
    avg_revenue_per_store = category_data["avg_revenue"]
    population_estimate = random.randint(100000, 1000000)
    ideal_stores_per_capita = 0.0001
    ideal_store_count = population_estimate * ideal_stores_per_capita
    market_saturation = (
        min((total_competitors / ideal_store_count) * 100, 100)
        if ideal_store_count > 0
        else 0
    )
    opportunity_score = max(0, (100 - market_saturation) * (growth_rate / 10))

    return MarketMetrics(
        market_size=market_size,
        growth_rate=growth_rate,
        competition_density=competition_density,
        avg_revenue_per_store=avg_revenue_per_store,
        market_saturation=market_saturation,
        opportunity_score=opportunity_score,
    )


def analyze_investment_potential(
    market_metrics: MarketMetrics, category_data: Dict, competitors: List[Dict]
) -> InvestmentAnalysis:
    startup_cost = category_data["startup_cost"]
    projected_annual_revenue = category_data["avg_revenue"]
    annual_profit_margin = 0.15
    annual_profit = projected_annual_revenue * annual_profit_margin
    roi_projection = (annual_profit / startup_cost) * 100
    payback_period = startup_cost / annual_profit
    risk_factors = []
    risk_score = 0

    if market_metrics.market_saturation > 70:
        risk_score += 30
        risk_factors.append("High market saturation")
    if market_metrics.competition_density > 10:
        risk_score += 20
        risk_factors.append("High competition density")
    if market_metrics.growth_rate < 5:
        risk_score += 25
        risk_factors.append("Low market growth rate")

    if roi_projection > 20 and risk_score < 40:
        recommendation = "Strong Buy"
        confidence = 85
    elif roi_projection > 15 and risk_score < 60:
        recommendation = "Buy"
        confidence = 70
    elif roi_projection > 10:
        recommendation = "Hold/Consider"
        confidence = 55
    else:
        recommendation = "Avoid"
        confidence = 30

    key_factors = [
        f"Market opportunity score: {market_metrics.opportunity_score:.1f}",
        f"Competition density: {market_metrics.competition_density} stores",
        f"Market growth rate: {market_metrics.growth_rate:.1f}%",
        f"Projected ROI: {roi_projection:.1f}%",
    ]

    return InvestmentAnalysis(
        roi_projection=roi_projection,
        payback_period=payback_period,
        risk_score=risk_score,
        investment_recommendation=recommendation,
        confidence_level=confidence,
        key_factors=key_factors,
    )


def integrate_agent_outputs(
    competitor_data: Dict,
    market_metrics: MarketMetrics,
    investment_analysis: InvestmentAnalysis,
) -> Dict[str, Any]:
    integrated_output = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "comprehensive_market_investment_analysis",
        "location": competitor_data.get("location", "Unknown"),
        "category": competitor_data.get("category", "Unknown"),
        "summary": {
            "total_competitors": len(competitor_data.get("results", [])),
            "market_size_usd": market_metrics.market_size,
            "investment_recommendation": investment_analysis.investment_recommendation,
            "confidence_level": investment_analysis.confidence_level,
            "opportunity_score": market_metrics.opportunity_score,
        },
        "detailed_analysis": {
            "market_metrics": {
                "size": market_metrics.market_size,
                "growth_rate": market_metrics.growth_rate,
                "saturation": market_metrics.market_saturation,
                "competition_density": market_metrics.competition_density,
            },
            "investment_metrics": {
                "roi_projection": investment_analysis.roi_projection,
                "payback_period": investment_analysis.payback_period,
                "risk_score": investment_analysis.risk_score,
                "key_factors": investment_analysis.key_factors,
            },
            "competitor_insights": competitor_data.get("results", []),
        },
        "recommendations": generate_strategic_recommendations(
            market_metrics, investment_analysis, competitor_data
        ),
    }
    return integrated_output


def generate_strategic_recommendations(
    market_metrics: MarketMetrics,
    investment_analysis: InvestmentAnalysis,
    competitor_data: Dict,
) -> List[str]:
    recommendations = []
    if market_metrics.opportunity_score > 70:
        recommendations.append(
            "🚀 High opportunity market - Consider aggressive expansion strategy"
        )
    elif market_metrics.opportunity_score > 40:
        recommendations.append(
            "📈 Moderate opportunity - Proceed with careful market entry"
        )
    else:
        recommendations.append(
            "⚠️ Limited opportunity - Consider alternative markets or strategies"
        )

    if investment_analysis.roi_projection > 20:
        recommendations.append("💰 Strong ROI potential - Prioritize this investment")
    elif investment_analysis.payback_period < 3:
        recommendations.append("⏱️ Quick payback period - Good cash flow investment")

    if market_metrics.market_saturation < 50:
        recommendations.append(
            "🎯 Undersaturated market - First-mover advantage possible"
        )

    competitors = competitor_data.get("results", [])
    if competitors:
        top_competitor = max(competitors, key=lambda x: x.get("footfall_daily_avg", 0))
        recommendations.append(
            f"🏆 Monitor top competitor: {top_competitor['name']} - {top_competitor['footfall_daily_avg']} daily footfall"
        )

    return recommendations


# LangGraph Agent Nodes
def competitor_node(state: AgentState) -> AgentState:
    try:
        coords = geocode_location(state["location"])
        category_data = map_query_to_category_data(state["query"])
        elements = query_overpass(coords["lat"], coords["lon"], category_data["tag"])
        competitors = analyze_competitors(
            elements, coords["lat"], coords["lon"], category_data
        )

        state["competitor_data"] = {
            "location": state["location"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": category_data["tag"],
            "results": competitors,
        }
        state["messages"].append("Competitor analysis completed")
        state["iteration_count"] += 1
        return state
    except Exception as e:
        logger.error(f"Error in competitor node: {e}")
        state["messages"].append(f"Competitor analysis failed: {str(e)}")
        return state


def market_node(state: AgentState) -> AgentState:
    try:
        competitors = state["competitor_data"]["results"]
        category_data = map_query_to_category_data(state["query"])
        market_metrics = analyze_market_metrics(
            competitors, category_data, state["location"]
        )
        state["market_data"] = vars(market_metrics)
        state["messages"].append("Market analysis completed")
        state["iteration_count"] += 1
        return state
    except Exception as e:
        logger.error(f"Error in market node: {e}")
        state["messages"].append(f"Market analysis failed: {str(e)}")
        return state


def investment_node(state: AgentState) -> AgentState:
    try:
        competitors = state["competitor_data"]["results"]
        category_data = map_query_to_category_data(state["query"])
        market_metrics = MarketMetrics(**state["market_data"])
        investment_analysis = analyze_investment_potential(
            market_metrics, category_data, competitors
        )
        state["investor_analysis"] = vars(investment_analysis)
        state["messages"].append("Investment analysis completed")
        state["iteration_count"] += 1
        return state
    except Exception as e:
        logger.error(f"Error in investment node: {e}")
        state["messages"].append(f"Investment analysis failed: {str(e)}")
        return state


def integration_node(state: AgentState) -> AgentState:
    try:
        market_metrics = MarketMetrics(**state["market_data"])
        investment_analysis = InvestmentAnalysis(**state["investor_analysis"])
        comprehensive_analysis = integrate_agent_outputs(
            state["competitor_data"], market_metrics, investment_analysis
        )
        state["agent_outputs"].append(comprehensive_analysis)
        state["messages"].append("Analysis integration completed")
        state["iteration_count"] += 1
        return state
    except Exception as e:
        logger.error(f"Error in integration node: {e}")
        state["messages"].append(f"Integration failed: {str(e)}")
        return state


# LangGraph Workflow Setup
def create_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("competitor", competitor_node)
    workflow.add_node("market", market_node)
    workflow.add_node("investment", investment_node)
    workflow.add_node("integration", integration_node)

    workflow.set_entry_point("competitor")
    workflow.add_edge("competitor", "market")
    workflow.add_edge("market", "investment")
    workflow.add_edge("investment", "integration")
    workflow.add_edge("integration", END)

    return workflow.compile()


# Generate comprehensive report (unchanged)
def generate_comprehensive_report(analysis: Dict[str, Any]) -> str:
    if "error" in analysis:
        return f"❌ Error generating analysis: {analysis['error']}"

    summary = analysis["summary"]
    market_metrics = analysis["detailed_analysis"]["market_metrics"]
    investment_metrics = analysis["detailed_analysis"]["investment_metrics"]
    competitors = analysis["detailed_analysis"]["competitor_insights"]
    recommendations = analysis["recommendations"]

    report = f"""
# 📊 Comprehensive Market & Investment Analysis
**Location:** {analysis['location']} | **Category:** {analysis['category']} | **Generated:** {analysis['timestamp'][:19]}

## 🎯 Executive Summary
- **Total Competitors:** {summary['total_competitors']}
- **Market Size:** ${summary['market_size_usd']:,.0f}
- **Investment Recommendation:** {summary['investment_recommendation']}
- **Confidence Level:** {summary['confidence_level']}%
- **Opportunity Score:** {summary['opportunity_score']:.1f}/100

## 📈 Market Analysis
- **Market Growth Rate:** {market_metrics['growth_rate']:.1f}% annually
- **Market Saturation:** {market_metrics['saturation']:.1f}%
- **Competition Density:** {market_metrics['competition_density']} competitors within 5km
- **Average Revenue per Store:** ${market_metrics['size']/max(1, market_metrics['competition_density']):,.0f}

## 💼 Investment Analysis
- **Projected ROI:** {investment_metrics['roi_projection']:.1f}%
- **Payback Period:** {investment_metrics['payback_period']:.1f} years
- **Risk Score:** {investment_metrics['risk_score']}/100
- **Key Investment Factors:**
"""

    for factor in investment_metrics["key_factors"]:
        report += f"  - {factor}\n"

    report += "\n## 🏪 Top Competitors Analysis\n"

    for i, comp in enumerate(competitors[:5], 1):
        report += f"""
**{i}. {comp['name']}**
- Distance: {comp['distance_km']} km
- Daily Footfall: {comp['footfall_daily_avg']}
- Est. Monthly Revenue: ${comp['estimated_monthly_revenue']:,}
- Market Share: {comp['market_share_estimate']}%
- Competitive Strength: {comp['competitive_strength']}
- Peak Hours: {', '.join(comp['peak_hours'])}
"""

    report += "\n## 🎯 Strategic Recommendations\n"
    for rec in recommendations:
        report += f"- {rec}\n"

    report += f"""
---
*This analysis integrates competitor intelligence, market dynamics, and investment potential to provide comprehensive business insights.*
"""

    return report


# Updated Streamlit UI with LangGraph integration
def main():
    st.set_page_config(
        page_title="AI Market & Investment Analyst", page_icon="📊", layout="wide"
    )

    st.title("🤖 AI-Powered Market & Investment Analyst")
    st.markdown(
        "*Comprehensive competitor analysis, market intelligence, and investment recommendations powered by multi-agent AI*"
    )

    # Sidebar for inputs
    with st.sidebar:
        st.header("Analysis Parameters")
        location = st.text_input(
            "📍 Location",
            value="Coimbatore, RS Puram",
            help="Enter city, area, or specific location",
        )
        query = st.text_input(
            "🏪 Business Category",
            value="clothing stores",
            help="e.g., 'restaurants', 'cafes', 'clothing stores'",
        )
        analysis_type = st.selectbox(
            "Analysis Type",
            [
                "Comprehensive Analysis",
                "Competitor Focus",
                "Market Focus",
                "Investment Focus",
            ],
        )

        if st.button("🚀 Generate Analysis", type="primary"):
            if not location or not query:
                st.error("Please enter both location and business category.")
                return

            with st.spinner("🔄 Running multi-agent analysis..."):
                workflow = create_workflow()
                initial_state = {
                    "messages": [],
                    "competitor_data": {},
                    "market_data": {},
                    "investor_analysis": {},
                    "location": location,
                    "query": query,
                    "iteration_count": 0,
                    "agent_outputs": [],
                }
                result = workflow.invoke(initial_state)
                if result["agent_outputs"]:
                    st.session_state["analysis"] = result["agent_outputs"][-1]
                else:
                    st.session_state["analysis"] = {
                        "error": "Analysis failed",
                        "messages": result["messages"],
                    }

    # Main content area (unchanged)
    if "analysis" in st.session_state:
        analysis = st.session_state["analysis"]

        if "error" not in analysis:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Competitors", analysis["summary"]["total_competitors"])
            with col2:
                st.metric(
                    "Market Size", f"${analysis['summary']['market_size_usd']:,.0f}"
                )
            with col3:
                st.metric(
                    "Opportunity Score",
                    f"{analysis['summary']['opportunity_score']:.1f}/100",
                )
            with col4:
                st.metric(
                    "Confidence Level", f"{analysis['summary']['confidence_level']}%"
                )

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "📊 Overview",
                    "🏪 Competitors",
                    "📈 Market Analysis",
                    "💼 Investment Analysis",
                ]
            )

            with tab1:
                st.markdown("## 📋 Comprehensive Report")
                report = generate_comprehensive_report(analysis)
                st.markdown(report)
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"market_analysis_{analysis['location'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                )

            with tab2:
                st.markdown("## 🏪 Competitor Intelligence")
                competitors = analysis["detailed_analysis"]["competitor_insights"]

                if competitors:
                    df = pd.DataFrame(competitors)
                    st.dataframe(df, use_container_width=True)
                    fig = px.scatter(
                        df,
                        x="distance_km",
                        y="footfall_daily_avg",
                        size="estimated_monthly_revenue",
                        hover_name="name",
                        title="Competitor Footfall vs Distance",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No competitors found in the specified area.")

            with tab3:
                st.markdown("## 📈 Market Intelligence")
                market_data = analysis["detailed_analysis"]["market_metrics"]
                fig = go.Figure()
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=market_data["saturation"],
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Market Saturation %"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 80], "color": "gray"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90,
                            },
                        },
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.markdown("## 💼 Investment Intelligence")
                investment_data = analysis["detailed_analysis"]["investment_metrics"]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "ROI Projection", f"{investment_data['roi_projection']:.1f}%"
                    )
                    st.metric(
                        "Payback Period",
                        f"{investment_data['payback_period']:.1f} years",
                    )

                with col2:
                    st.metric("Risk Score", f"{investment_data['risk_score']}/100")
                    recommendation = analysis["summary"]["investment_recommendation"]
                    color = (
                        "green"
                        if "Buy" in recommendation
                        else "orange" if "Hold" in recommendation else "red"
                    )
                    st.markdown(f"**Recommendation:** :{color}[{recommendation}]")

        else:
            st.error(f"Analysis failed: {analysis['error']}")

    else:
        st.info(
            "👈 Enter your analysis parameters in the sidebar and click 'Generate Analysis' to begin."
        )
        st.markdown("## 🎯 What This Tool Provides")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            **🏪 Competitor Analysis**
            - Real competitor locations
            - Footfall estimates
            - Revenue projections
            - Market share analysis
            """
            )

        with col2:
            st.markdown(
                """
            **📈 Market Intelligence**
            - Market size calculation
            - Growth rate analysis
            - Saturation metrics
            - Opportunity scoring
            """
            )

        with col3:
            st.markdown(
                """
            **💼 Investment Analysis**
            - ROI projections
            - Risk assessment
            - Payback calculations
            - Strategic recommendations
            """
            )


if __name__ == "__main__":
    main()
