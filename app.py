import re
from typing import Dict, Iterable, List

import pandas as pd
import streamlit as st


DATA_PATH = "data/creators.csv"
REQUIRED_COLUMNS = {
    "creator_name",
    "niche",
    "region",
    "avg_views",
    "engagement_rate",
    "audience_age",
    "content_style",
    "brand_safety_score",
    "past_campaign_type",
    "estimated_cost",
}


st.set_page_config(
    page_title="CreatorMatch AI",
    page_icon="CM",
    layout="wide",
    initial_sidebar_state="expanded",
)


def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def tokens(value: object) -> set:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


@st.cache_data(show_spinner=False)
def load_default_creators() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_creators(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_default_creators()


def validate_creators(df: pd.DataFrame) -> List[str]:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}"]

    numeric_columns = ["avg_views", "engagement_rate", "brand_safety_score", "estimated_cost"]
    errors = []
    for column in numeric_columns:
        if pd.to_numeric(df[column], errors="coerce").isna().any():
            errors.append(f"Column `{column}` must contain numeric values.")
    return errors


def keyword_fit(creator_values: Iterable[object], brief_values: Iterable[object]) -> float:
    creator_tokens = set()
    for value in creator_values:
        creator_tokens.update(tokens(value))

    brief_tokens = set()
    for value in brief_values:
        brief_tokens.update(tokens(value))

    if not brief_tokens:
        return 50.0

    overlap = creator_tokens & brief_tokens
    direct_overlap_score = min(len(overlap) / max(len(brief_tokens), 1), 1.0) * 100

    phrase_bonus = 0
    creator_text = " ".join(normalize_text(value) for value in creator_values)
    for value in brief_values:
        text = normalize_text(value)
        if text and text in creator_text:
            phrase_bonus += 20

    return clamp(direct_overlap_score + phrase_bonus)


def parse_age_range(value: object) -> tuple:
    numbers = [int(number) for number in re.findall(r"\d+", str(value))]
    if not numbers:
        return (0, 100)
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    return (min(numbers), max(numbers))


def audience_fit(creator_age: object, target_audience: str, creator_region: str, target_region: str) -> float:
    creator_min, creator_max = parse_age_range(creator_age)
    audience_min, audience_max = parse_age_range(target_audience)

    overlap = max(0, min(creator_max, audience_max) - max(creator_min, audience_min))
    audience_span = max(1, audience_max - audience_min)
    age_score = clamp((overlap / audience_span) * 100)

    if not re.search(r"\d+", target_audience):
        age_score = keyword_fit([creator_age], [target_audience])

    region_score = 100 if normalize_text(target_region) in {"", "global"} else 0
    if normalize_text(creator_region) == normalize_text(target_region):
        region_score = 100
    elif normalize_text(target_region) in normalize_text(creator_region) or normalize_text(creator_region) in normalize_text(target_region):
        region_score = 75

    return clamp((age_score * 0.7) + (region_score * 0.3))


def budget_fit(creator_cost: float, budget: float) -> float:
    if budget <= 0:
        return 50.0
    if creator_cost <= budget:
        return 100.0
    overage_ratio = (creator_cost - budget) / budget
    return clamp(100 - (overage_ratio * 100))


def score_creators(df: pd.DataFrame, brief: Dict[str, object]) -> pd.DataFrame:
    scored = df.copy()
    for column in ["avg_views", "engagement_rate", "brand_safety_score", "estimated_cost"]:
        scored[column] = pd.to_numeric(scored[column], errors="coerce")

    max_engagement = max(scored["engagement_rate"].max(), 1)

    niche_scores = []
    audience_scores = []
    engagement_scores = []
    safety_scores = []
    budget_scores = []

    for _, row in scored.iterrows():
        niche_scores.append(
            keyword_fit(
                [row["niche"], row["past_campaign_type"], row["content_style"]],
                [brief["product_category"], brief["campaign_goal"], brief["product_benefit"]],
            )
        )
        audience_scores.append(
            audience_fit(row["audience_age"], brief["target_audience"], row["region"], brief["region"])
        )
        engagement_scores.append(clamp((row["engagement_rate"] / max_engagement) * 100))
        safety_scores.append(clamp(row["brand_safety_score"]))
        budget_scores.append(budget_fit(row["estimated_cost"], brief["budget"]))

    scored["niche_fit"] = niche_scores
    scored["audience_fit"] = audience_scores
    scored["engagement_fit"] = engagement_scores
    scored["safety_fit"] = safety_scores
    scored["budget_fit"] = budget_scores

    scored["match_score"] = (
        scored["niche_fit"] * 0.30
        + scored["audience_fit"] * 0.25
        + scored["engagement_fit"] * 0.20
        + scored["safety_fit"] * 0.15
        + scored["budget_fit"] * 0.10
    ).round(1)

    return scored.sort_values("match_score", ascending=False).reset_index(drop=True)


def creator_format_strategy(row: pd.Series) -> Dict[str, str]:
    niche = normalize_text(row["niche"])
    style = normalize_text(row["content_style"])
    profile = f"{niche} {style}"

    if any(word in profile for word in ["college", "student", "dorm", "humor", "skit"]):
        return {
            "angle": "Build a class-crash or dorm-life skit where the product naturally solves a student moment.",
            "hook": "Open on a familiar campus problem, then let the product become the punchline or quick fix.",
            "format": "relatable student humor",
        }
    if any(word in profile for word in ["fitness", "workout", "gym", "performance"]):
        return {
            "angle": "Frame the product inside a pre-gym, workout routine, recovery, or performance-prep moment.",
            "hook": "Start with a quick routine setup, then show where the product fits before, during, or after movement.",
            "format": "energetic routine demo",
        }
    if any(word in profile for word in ["food", "taste", "snack", "restaurant"]):
        return {
            "angle": "Use a taste-test reaction with a snack, meal-pairing, or first-bite reveal.",
            "hook": "Lead with the reaction moment, then explain why the product earns a spot in the meal or snack rotation.",
            "format": "taste-test reaction",
        }
    if any(word in profile for word in ["beauty", "skincare", "grwm", "routine"]):
        return {
            "angle": "Turn the product into a GRWM, before-class routine, or quick transformation step.",
            "hook": "Begin mid-routine with a visible product moment, then show how it completes the look or ritual.",
            "format": "routine-led beauty tutorial",
        }
    if any(word in profile for word in ["tech", "review", "explainer", "app", "software"]):
        return {
            "angle": "Create a feature breakdown, side-by-side comparison, or problem-to-solution explainer.",
            "hook": "Start with the user pain point, then walk through the feature that makes the product easier to understand.",
            "format": "clear product explainer",
        }
    if any(word in profile for word in ["fashion", "outfit", "style", "transition"]):
        return {
            "angle": "Use outfit transitions to connect the product to a lifestyle aesthetic or occasion-based look.",
            "hook": "Open with the final look, rewind into the transition, and show where the brand fits in the styling story.",
            "format": "outfit transition",
        }
    if any(word in profile for word in ["gaming", "live", "reaction", "play"]):
        return {
            "angle": "Place the product in a gaming-session setup, long-play focus moment, or live reaction beat.",
            "hook": "Start during an intense play moment, then connect the product to the session setup or creator routine.",
            "format": "gaming reaction",
        }
    if any(word in profile for word in ["wellness", "calm", "mindful", "health"]):
        return {
            "angle": "Show the product as part of a daily routine, balance reset, or mindful lifestyle ritual.",
            "hook": "Open with a calm reset moment, then show how the product supports a simple habit.",
            "format": "calm lifestyle routine",
        }

    return {
        "angle": f"Use the creator's {row['content_style']} style to translate the product benefit into a native story.",
        "hook": "Open with a specific audience tension, then demonstrate the product in the creator's usual format.",
        "format": str(row["content_style"]),
    }


def recommend_kpis(goal: str, row: pd.Series = None) -> tuple:
    goal_text = normalize_text(goal)
    format_text = normalize_text(row["content_style"]) if row is not None else ""

    if any(word in goal_text for word in ["creator", "ugc", "testing", "test", "creative"]):
        secondary = "Hook retention" if any(word in format_text for word in ["skit", "reaction", "transition"]) else "Save/share rate"
        return "Engagement rate", secondary
    if any(word in goal_text for word in ["conversion", "sales", "purchase", "purchases"]):
        secondary = "ROAS" if "shop" in goal_text or "purchase" in goal_text else "CPA"
        return "Conversion rate", secondary
    if any(word in goal_text for word in ["traffic", "site", "app", "landing"]):
        return "CTR", "Landing page visits"
    if any(word in goal_text for word in ["awareness", "reach", "launch"]):
        secondary = "Engagement rate" if any(word in format_text for word in ["skit", "reaction", "grwm"]) else "Video views"
        return "Reach", secondary
    if any(word in goal_text for word in ["engagement", "community", "ugc"]):
        return "Engagement rate", "Save/share rate"
    return "Engagement rate", "Video views"


def risk_flag(row: pd.Series, brief: Dict[str, object]) -> str:
    flags = []
    product_context = " ".join(
        normalize_text(value)
        for value in [
            brief["product_category"],
            brief["product_benefit"],
            brief["campaign_goal"],
        ]
    )

    if row["brand_safety_score"] < 88:
        flags.append("Brand safety review recommended before approval")
    if brief["budget"] > 0 and row["estimated_cost"] >= brief["budget"] * 0.9:
        flags.append("Budget efficiency risk because estimated cost is near or above budget")
    if row.get("audience_fit", 100) < 50:
        flags.append("Audience fit needs validation against target age or region")
    if any(word in product_context for word in ["fitness", "wellness", "energy drink", "energy", "performance"]):
        flags.append("Avoid exaggerated health, energy, or performance claims")
    if any(word in product_context for word in ["finance", "fintech", "financial", "money", "invest", "credit"]):
        flags.append("Avoid misleading financial claims or guaranteed outcome language")
    if any(word in product_context for word in ["beauty", "skincare", "skin", "acne", "glow"]):
        flags.append("Avoid unrealistic before/after or results claims")

    if flags:
        return "; ".join(dict.fromkeys(flags))
    return f"Low risk: strong safety score ({int(row['brand_safety_score'])}/100) and cost within plan"


def creative_angle(row: pd.Series, brief: Dict[str, object]) -> str:
    benefit = brief["product_benefit"] or "the core product benefit"
    strategy = creator_format_strategy(row)
    return (
        f"{strategy['angle']} For {brief['brand_name']}, connect the {row['niche']} audience to "
        f"{benefit} through {row['content_style']}."
    )


def fit_reason(row: pd.Series, brief: Dict[str, object]) -> str:
    budget_note = "within budget" if row["estimated_cost"] <= brief["budget"] else "above the current budget"
    safety_note = "strong brand-safety profile" if row["brand_safety_score"] >= 90 else "brand-safety score that should be reviewed"
    return (
        f"{row['creator_name']} brings {row['niche']} credibility through {row['content_style']} content, "
        f"with an audience age range of {row['audience_age']} and {row['engagement_rate']}% engagement. "
        f"Their past campaign experience in {row['past_campaign_type']} gives useful context, while the "
        f"${int(row['estimated_cost']):,} estimate is {budget_note} and the {int(row['brand_safety_score'])}/100 "
        f"score indicates a {safety_note}."
    )


def fallback_creator_brief(row: pd.Series, brief: Dict[str, object]) -> str:
    primary_kpi, secondary_kpi = recommend_kpis(brief["campaign_goal"], row)
    strategy = creator_format_strategy(row)
    return (
        f"Creator brief for {row['creator_name']}: Make a {brief['tone'].lower()} TikTok for "
        f"{brief['brand_name']} that feels native to your {row['niche']} community and your "
        f"{row['content_style']} format. {strategy['hook']} Show the product benefit, "
        f"{brief['product_benefit']}, through a real moment your {row['audience_age']} audience would recognize. "
        f"Reference the kind of category familiarity you have from {row['past_campaign_type']} work, keep the brand "
        f"mention conversational, and close with one clear next step. Optimize for {primary_kpi}; use "
        f"{secondary_kpi} to judge whether the format is worth scaling."
    )


def generate_creator_brief(row: pd.Series, brief: Dict[str, object]) -> str:
    return fallback_creator_brief(row, brief)


def render_creator_card(row: pd.Series, brief: Dict[str, object]) -> None:
    primary_kpi, secondary_kpi = recommend_kpis(brief["campaign_goal"], row)
    st.markdown(f"#### #{int(row.name) + 1} {row['creator_name']}")

    st.write(f"**Why this creator fits:** {fit_reason(row, brief)}")
    st.write(f"**Recommended creative angle:** {creative_angle(row, brief)}")

    kpi_cols = st.columns(3)
    kpi_cols[0].write(f"**Primary KPI:** {primary_kpi}")
    kpi_cols[1].write(f"**Secondary KPI:** {secondary_kpi}")
    kpi_cols[2].write(f"**Risk flag:** {risk_flag(row, brief)}")

    with st.expander("Creator-ready brief", expanded=True):
        st.write(generate_creator_brief(row, brief))


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
    div[data-testid="stExpander"] {
        border: 1px solid #e7eaf0;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("CreatorMatch AI")
st.caption("Campaign brief to ranked creator recommendations, creative angles, KPIs, safety flags, and creator-ready briefs.")

st.info(
    "This demo uses a synthetic creator dataset for safe public portfolio use. "
    "You can upload your own creator CSV in the sidebar to test the same workflow with different data.",
)

with st.sidebar:
    st.header("Creator Data")
    uploaded_file = st.file_uploader("Upload creator CSV", type=["csv"])
    st.caption("Expected columns: " + ", ".join(sorted(REQUIRED_COLUMNS)))

try:
    creators = load_creators(uploaded_file)
except Exception as exc:
    st.error(f"Could not load creator data: {exc}")
    st.stop()

validation_errors = validate_creators(creators)
if validation_errors:
    for error in validation_errors:
        st.error(error)
    st.stop()

with st.form("campaign_brief"):
    st.subheader("Campaign Brief")
    left, right = st.columns(2)
    with left:
        brand_name = st.text_input("Brand name", value="GlowLab")
        product_category = st.text_input("Product category", value="skincare beauty")
        campaign_goal = st.text_input("Campaign goal", value="drive awareness for a product launch")
        target_audience = st.text_input("Target audience", value="18-30 college students")
    with right:
        tone = st.selectbox("Tone", ["Playful", "Educational", "Aspirational", "Bold", "Calm"], index=0)
        region = st.selectbox("Region", sorted(creators["region"].dropna().unique().tolist()), index=0)
        budget = st.number_input("Budget per creator", min_value=0, value=1800, step=100)
        product_benefit = st.text_input("Product benefit", value="simple routines that make skin feel fresh")

    submitted = st.form_submit_button("Find creator matches", type="primary")

brief = {
    "brand_name": brand_name,
    "product_category": product_category,
    "campaign_goal": campaign_goal,
    "target_audience": target_audience,
    "tone": tone,
    "region": region,
    "budget": float(budget),
    "product_benefit": product_benefit,
}

if submitted:
    ranked = score_creators(creators, brief)
    top_creators = ranked.head(5).copy()

    st.subheader("Top Creator Matches")
    st.dataframe(
        top_creators[
            [
                "creator_name",
                "match_score",
                "niche",
                "region",
                "audience_age",
                "engagement_rate",
                "brand_safety_score",
                "estimated_cost",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Recommendations")
    for display_index, (_, creator_row) in enumerate(top_creators.iterrows()):
        creator_row = creator_row.copy()
        creator_row.name = display_index
        render_creator_card(creator_row, brief)
        st.divider()
else:
    st.subheader("How the demo works")
    st.write(
        "Fill out the campaign brief and CreatorMatch AI will rank creators using weighted fit across niche, "
        "audience, engagement, brand safety, and budget. The recommendation details use rule-based logic so "
        "the demo works consistently without external APIs."
    )

    preview_cols = st.columns(4)
    preview_cols[0].metric("Creators loaded", len(creators))
    preview_cols[1].metric("Avg. engagement", f"{creators['engagement_rate'].mean():.1f}%")
    preview_cols[2].metric("Avg. safety", f"{creators['brand_safety_score'].mean():.0f}/100")
    preview_cols[3].metric("Median cost", f"${int(creators['estimated_cost'].median()):,}")
