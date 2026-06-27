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

    if any(word in profile for word in ["humor", "skit", "funny"]):
        return {
            "angle": "Build a class-crash or dorm-life skit where the product naturally solves a student moment.",
            "hook": "Open on a familiar campus problem, then let the product become the punchline or quick fix.",
            "format": "relatable student humor",
        }
    if any(word in profile for word in ["college", "student", "dorm", "campus", "day in the life"]):
        return {
            "angle": "Place the product inside a campus routine, dorm reset, study session, or day-in-the-life beat.",
            "hook": "Start inside the creator's daily campus rhythm, then show where the product fits without breaking the routine.",
            "format": "college lifestyle routine",
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
    niche_text = normalize_text(row["niche"]) if row is not None else ""
    profile = f"{niche_text} {format_text}"

    if any(word in goal_text for word in ["conversion", "sales", "purchase", "purchases"]):
        secondary = "CPA/ROAS" if "shop" in goal_text or "purchase" in goal_text else "CPA"
        return "Conversion rate", secondary
    if any(word in goal_text for word in ["awareness", "reach", "launch"]):
        return "Video views", "Reach"
    if any(word in goal_text for word in ["creator", "ugc", "testing", "test", "creative"]):
        secondary = "Shares" if any(word in profile for word in ["skit", "reaction", "humor"]) else "Save/share rate"
        return "Engagement rate", secondary
    if any(word in goal_text for word in ["traffic", "site", "app", "landing"]):
        return "CTR", "Landing page visits"
    if any(word in goal_text for word in ["engagement", "community", "ugc"]):
        return "Engagement rate", "Save/share rate"

    if any(word in profile for word in ["skit", "reaction", "humor", "taste", "gaming"]):
        return "Engagement rate", "Shares/CTR"
    if any(word in profile for word in ["demo", "fitness", "product"]):
        return "Conversion rate", "CPA/ROAS"
    if any(word in profile for word in ["explainer", "review", "tech", "finance"]):
        return "CTR", "Landing-page visits/saves"
    if any(word in profile for word in ["lifestyle", "routine", "grwm", "transition", "wellness"]):
        return "Saves", "Completion rate"
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
    direction = creative_pack_direction(row)
    return (
        f"Creator brief for {row['creator_name']}: Produce a {brief['tone'].lower()} TikTok in your "
        f"{row['content_style']} format, shaped as a {direction['archetype']} for your {row['niche']} community. "
        f"{strategy['hook']} Bring in {brief['brand_name']} as the natural answer to {brief['product_benefit']} "
        f"for viewers around {row['audience_age']}. Use your past {row['past_campaign_type']} experience as context, "
        f"keep the claim language appropriate for a {int(row['brand_safety_score'])}/100 safety profile, and make the "
        f"${int(row['estimated_cost']):,} creator fee work by focusing on one memorable product moment. Optimize for "
        f"{primary_kpi}; use {secondary_kpi} and the creator's {row['engagement_rate']}% engagement rate to judge "
        f"whether the format is worth scaling."
    )


def generate_creator_brief(row: pd.Series, brief: Dict[str, object]) -> str:
    return fallback_creator_brief(row, brief)


def tone_directive(tone: str) -> str:
    tone_text = normalize_text(tone)
    directives = {
        "playful": "light, punchy, and conversational",
        "educational": "clear, useful, and proof-led",
        "aspirational": "polished, confident, and lifestyle-led",
        "bold": "direct, high-energy, and scroll-stopping",
        "calm": "grounded, warm, and reassuring",
    }
    return directives.get(tone_text, f"{tone} and native to TikTok")


def audience_label(target_audience: str, creator_age: object) -> str:
    if target_audience:
        return target_audience
    return f"{creator_age} viewers"


def creative_pack_direction(row: pd.Series) -> Dict[str, object]:
    niche = normalize_text(row["niche"])
    style = normalize_text(row["content_style"])
    profile = f"{niche} {style}"

    if any(word in profile for word in ["humor", "funny", "skit"]):
        return {
            "archetype": "comedy skit",
            "hook_set": [
                "POV: the group project is due tonight and everyone is running on fumes",
                "That 3 PM crash before finals when the dorm becomes a survival show",
                "When class starts in five minutes and your routine needs a cheat code",
            ],
            "caption_voice": "campus-joke energy with a clear product payoff",
            "thumbnail_scene": "student life, dorm room chaos, textbooks, classmates, and a punchline-ready expression",
            "storyboard": [
                ("0-3s", "Cold open on a class, finals, or group-project problem with a quick comedic reaction."),
                ("3-6s", "Cut to the creator in dorm-life skit mode introducing the product as the unexpected fix."),
                ("6-10s", "Escalate the joke with classmates reacting while the product benefit lands naturally."),
                ("10-13s", "Show the before/after energy of the moment without making unrealistic claims."),
                ("13-15s", "End on a punchline, brand frame, and share-friendly CTA."),
            ],
            "cta_style": "Tag the friend who hits the 3 PM crash first",
        }
    if any(word in profile for word in ["fitness", "workout", "gym", "performance", "demo"]):
        return {
            "archetype": "workout/product demo",
            "hook_set": [
                "What I use before the gym when I want my routine to feel less chaotic",
                "Pre-workout checklist: bag, playlist, and the product I keep reaching for",
                "Active lifestyle test: does this actually fit into a real training day?",
            ],
            "caption_voice": "high-energy routine proof with a product demo",
            "thumbnail_scene": "gym bag, locker-room mirror, active routine setup, or mid-workout product moment",
            "storyboard": [
                ("0-3s", "Open with a gym-bag pack, warmup, or pre-workout routine beat."),
                ("3-6s", "Show the product entering the routine with one clear use case."),
                ("6-10s", "Demo the benefit during movement, prep, or recovery while keeping claims grounded."),
                ("10-13s", "Cut to the creator's quick verdict for an active audience."),
                ("13-15s", "Close with product in hand and a conversion-minded next step."),
            ],
            "cta_style": "Try it in your next routine and see if it earns a spot",
        }
    if any(word in profile for word in ["tech", "review", "app", "software"]):
        return {
            "archetype": "explainer/review",
            "hook_set": [
                "Is this actually worth it? I tested the feature people keep asking about",
                "Feature breakdown: the one thing that makes this product easier to use",
                "Side-by-side review: what changes after using this for a real task?",
            ],
            "caption_voice": "practical review language with a clean verdict",
            "thumbnail_scene": "desk setup, phone or laptop screen, product review frame, and comparison labels",
            "storyboard": [
                ("0-3s", "Open at a desk setup with the review question or comparison on screen."),
                ("3-6s", "Introduce the product and the exact feature being tested."),
                ("6-10s", "Walk through a simple before/after or side-by-side use case."),
                ("10-13s", "Give the creator's verdict tied to the campaign benefit."),
                ("13-15s", "Point viewers to learn more, click, or save the review."),
            ],
            "cta_style": "Save this review or tap through for the full breakdown",
        }
    if any(word in profile for word in ["college", "student", "dorm", "campus", "day in the life"]):
        return {
            "archetype": "lifestyle routine",
            "hook_set": [
                "Day in the life: the campus routine step I did not expect to keep",
                "Study session reset with the product that makes my dorm routine easier",
                "Come with me from class to library and see where this fits in",
            ],
            "caption_voice": "natural campus lifestyle narration",
            "thumbnail_scene": "campus walkway, dorm desk, backpack, study setup, or daily routine flat lay",
            "storyboard": [
                ("0-3s", "Start with a campus transition or dorm morning routine."),
                ("3-6s", "Place the product in a real study, class, or reset moment."),
                ("6-10s", "Show the routine benefit while the creator moves through the day."),
                ("10-13s", "Add a quick creator reflection on why it fits student life."),
                ("13-15s", "Close with a soft save-or-try CTA and clean brand frame."),
            ],
            "cta_style": "Save this for your next campus routine",
        }
    if any(word in profile for word in ["beauty", "skincare", "grwm"]):
        return {
            "archetype": "tutorial/routine",
            "hook_set": [
                "GRWM before class with the routine step I am keeping",
                "Getting ready in a rush: where this fits before I leave",
                "My quick routine when I want skin to feel fresh without overthinking it",
            ],
            "caption_voice": "beauty routine detail with a creator-to-camera feel",
            "thumbnail_scene": "vanity, mirror, skincare shelf, product texture, or getting-ready setup",
            "storyboard": [
                ("0-3s", "Open mid-GRWM at the mirror with the creator naming the routine moment."),
                ("3-6s", "Show product application or placement in the routine."),
                ("6-10s", "Cut between texture, mirror check, and the creator explaining the benefit."),
                ("10-13s", "Show the finished routine while avoiding exaggerated before/after claims."),
                ("13-15s", "Close with a beauty-community CTA and product shot."),
            ],
            "cta_style": "Add it to your next GRWM if your routine needs this step",
        }
    if any(word in profile for word in ["food", "taste", "snack", "restaurant"]):
        return {
            "archetype": "taste-test reaction",
            "hook_set": [
                "First bite reaction: does this belong in the snack rotation?",
                "Taste test with the pairing I did not expect to work this well",
                "Snack pairing check: would I bring this to the next hangout?",
            ],
            "caption_voice": "reaction-first food language",
            "thumbnail_scene": "first-bite expression, product next to a snack pairing, or split reaction frame",
            "storyboard": [
                ("0-3s", "Open on the first bite or sip reaction before explaining anything."),
                ("3-6s", "Reveal the product and the pairing or taste-test setup."),
                ("6-10s", "Give sensory notes and a fast creator reaction."),
                ("10-13s", "Connect the flavor or use occasion to the campaign benefit."),
                ("13-15s", "Close with a comment-or-share CTA for food discovery."),
            ],
            "cta_style": "Comment the pairing you would try with this",
        }
    if any(word in profile for word in ["finance", "fintech", "money", "budget"]):
        return {
            "archetype": "educational explainer",
            "hook_set": [
                "What I wish I knew before trying to budget smarter",
                "Smart spending check: the simple habit I would start with",
                "Budget-friendly breakdown: where this product can fit without the hype",
            ],
            "caption_voice": "clear financial education with careful, non-guaranteed language",
            "thumbnail_scene": "phone app screen, simple budget visual, calculator, notes, and clean finance labels",
            "storyboard": [
                ("0-3s", "Open with the money question or mistake the audience recognizes."),
                ("3-6s", "Introduce the product as a tool or habit support, not a guaranteed outcome."),
                ("6-10s", "Break down one practical use case with simple on-screen labels."),
                ("10-13s", "Summarize the takeaway and who it is best suited for."),
                ("13-15s", "Close with a save, learn-more, or compare CTA."),
            ],
            "cta_style": "Save this before your next budget reset",
        }
    if any(word in profile for word in ["fashion", "outfit", "style", "transition"]):
        return {
            "archetype": "outfit/lifestyle transition",
            "hook_set": [
                "Fit check: styling this around one product moment",
                "Outfit transition from basic to campaign-ready",
                "Aesthetic test: does this belong in the final look?",
            ],
            "caption_voice": "style-led copy with a visual transformation",
            "thumbnail_scene": "mirror shot, outfit transition frame, styled accessories, or before/after fit check",
            "storyboard": [
                ("0-3s", "Open on the final look, then snap back to the starting outfit."),
                ("3-6s", "Introduce the product as part of the styling decision."),
                ("6-10s", "Show two quick transitions or detail shots tied to the benefit."),
                ("10-13s", "Reveal the complete look in motion."),
                ("13-15s", "Close with a style CTA and brand/product frame."),
            ],
            "cta_style": "Save this fit idea for your next styling reset",
        }
    if any(word in profile for word in ["gaming", "live", "play"]):
        return {
            "archetype": "gaming reaction",
            "hook_set": [
                "Long session test: what stayed on my desk for squad night",
                "Live reaction after using this during a focus-heavy match",
                "Squad night setup check: does this help the session feel smoother?",
            ],
            "caption_voice": "creator reaction language for gaming sessions",
            "thumbnail_scene": "gaming setup, monitor glow, headset, desk gear, or live-reaction face cam",
            "storyboard": [
                ("0-3s", "Open on a tense gameplay or squad-night reaction."),
                ("3-6s", "Cut to the desk setup and introduce the product naturally."),
                ("6-10s", "Show how it fits during focus, breaks, or the creator's session routine."),
                ("10-13s", "Return to the reaction moment with a quick verdict."),
                ("13-15s", "Close with a community CTA for gamers."),
            ],
            "cta_style": "Drop this into your next squad-night setup",
        }
    if any(word in profile for word in ["wellness", "calm", "mindful", "health"]):
        return {
            "archetype": "calm routine",
            "hook_set": [
                "Slow morning reset with the product I would keep in the routine",
                "Night routine check: one small step for more balance",
                "Reset with me when the day needs to feel less rushed",
            ],
            "caption_voice": "soft routine language with a grounded benefit",
            "thumbnail_scene": "wellness shelf, nightstand, warm lighting, journal, or quiet lifestyle setup",
            "storyboard": [
                ("0-3s", "Open with a quiet morning, night, or reset ritual."),
                ("3-6s", "Introduce the product as one small step in the routine."),
                ("6-10s", "Show the creator using it while narrating the benefit calmly."),
                ("10-13s", "Hold on a simple lifestyle moment that feels achievable."),
                ("13-15s", "Close with a save-for-later CTA and gentle product frame."),
            ],
            "cta_style": "Save this for your next reset routine",
        }

    return {
        "archetype": str(row["content_style"]),
        "hook_set": [
            "The product moment I did not expect to fit this naturally",
            "Testing this in my usual routine so you do not have to guess",
            "A quick creator check before you decide if this is for you",
        ],
        "caption_voice": "creator-native TikTok copy",
        "thumbnail_scene": "authentic creator setup with product, face, and simple benefit cue",
        "storyboard": [
            ("0-3s", "Open with the creator's normal format and a clear audience tension."),
            ("3-6s", "Introduce the product in context."),
            ("6-10s", "Show one practical use case tied to the benefit."),
            ("10-13s", "Add the creator's verdict."),
            ("13-15s", "Close with a concise CTA."),
        ],
        "cta_style": "Check it out if this solves the same problem for you",
    }


def generate_aigc_creative_pack(row: pd.Series, brief: Dict[str, object]) -> Dict[str, object]:
    direction = creative_pack_direction(row)
    primary_kpi, secondary_kpi = recommend_kpis(brief["campaign_goal"], row)
    audience = audience_label(brief["target_audience"], row["audience_age"])
    tone = tone_directive(brief["tone"])
    brand = brief["brand_name"] or "the brand"
    category = brief["product_category"] or "the product"
    benefit = brief["product_benefit"] or "the product benefit"
    goal = brief["campaign_goal"] or "the campaign goal"
    niche = row["niche"]
    style = row["content_style"]
    creator = row["creator_name"]
    past_campaign = row["past_campaign_type"]
    engagement_rate = row["engagement_rate"]
    safety_score = int(row["brand_safety_score"])
    estimated_cost = int(row["estimated_cost"])
    format_label = direction["archetype"]

    hooks = [
        f"{direction['hook_set'][0]} - {creator} tests {brand} for {benefit}.",
        f"{direction['hook_set'][1]} - a {style} take for {audience}.",
        f"{direction['hook_set'][2]} - from a {niche} creator with {engagement_rate}% engagement.",
    ]

    caption = (
        f"{creator} brings a {format_label} spin to {brand}, using {style} to show {benefit} for "
        f"{audience}. Keep the voice {direction['caption_voice']} and {tone}. Built for {goal}; "
        f"direction inspired by prior {past_campaign} work and a "
        f"{engagement_rate}% engagement audience. #{normalize_text(category).replace(' ', '')} "
        f"#{normalize_text(niche).replace(' ', '')}"
    )

    thumbnail_prompt = (
        f"Vertical 9:16 image prompt for {creator}'s {format_label}: show {direction['thumbnail_scene']}. "
        f"Include {brand} as a {category} product, one visual cue for {benefit}, and styling that feels "
        f"{tone} for {audience}. Keep the integration brand-safe for a {safety_score}/100 creator profile, "
        f"leave room for a short headline, and avoid implying actual generated media or unrealistic claims."
    )

    storyboard = []
    for timestamp, beat in direction["storyboard"]:
        storyboard.append(
            (
                timestamp,
                (
                    f"{beat} Use {creator}'s {style} format, speak to {audience}, and ladder back to "
                    f"{brand}'s {benefit} benefit."
                ),
            )
        )

    cta = (
        f"{direction['cta_style']} with {brand}. Best for {primary_kpi} and {secondary_kpi}; estimated creator "
        f"cost is ${estimated_cost:,}, so keep the ask focused on one strong {format_label}."
    )

    return {
        "hooks": hooks,
        "caption": caption,
        "thumbnail_prompt": thumbnail_prompt,
        "storyboard": storyboard,
        "cta": cta,
    }


def render_aigc_creative_pack(row: pd.Series, brief: Dict[str, object]) -> None:
    creative_pack = generate_aigc_creative_pack(row, brief)

    with st.expander("AIGC Creative Pack", expanded=False):
        st.caption("Prompt and copy ideas only. This app does not generate actual images or videos.")

        hook_cols = st.columns(3)
        for index, hook in enumerate(creative_pack["hooks"], start=1):
            hook_cols[index - 1].write(f"**Hook {index}**")
            hook_cols[index - 1].write(hook)

        st.write(f"**TikTok caption:** {creative_pack['caption']}")
        st.write(f"**Image prompt:** {creative_pack['thumbnail_prompt']}")

        st.write("**15-second storyboard:**")
        storyboard_df = pd.DataFrame(
            creative_pack["storyboard"],
            columns=["Timestamp", "Beat"],
        )
        st.dataframe(storyboard_df, hide_index=True, use_container_width=True)

        st.write(f"**Creator CTA:** {creative_pack['cta']}")


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

    render_aigc_creative_pack(row, brief)


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
st.caption(
    "Campaign brief to ranked creator recommendations, creative angles, KPIs, safety flags, "
    "creator-ready briefs, and AIGC creative packs."
)

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
        "audience, engagement, brand safety, and budget. The recommendation details and AIGC creative packs "
        "use rule-based logic so the demo works consistently without external APIs."
    )

    preview_cols = st.columns(4)
    preview_cols[0].metric("Creators loaded", len(creators))
    preview_cols[1].metric("Avg. engagement", f"{creators['engagement_rate'].mean():.1f}%")
    preview_cols[2].metric("Avg. safety", f"{creators['brand_safety_score'].mean():.0f}/100")
    preview_cols[3].metric("Median cost", f"${int(creators['estimated_cost'].median()):,}")
