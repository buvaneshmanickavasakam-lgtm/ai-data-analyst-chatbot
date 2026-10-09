"""Question -> plan. Rule-based parser (always available) + optional LLM parser, both validated."""
import json
import re
import pandas as pd
from analysis.utils import split_tokens, stem, name_tokens, FREQ
from .prompt import ALLOWED_INTENTS, ALLOWED_AGGS, SYSTEM_INTENT, build_intent_prompt
from . import client

STOP = set("""the a an of in on for to is are was were what which who whom how does do did show me give tell about each with and or from
that this these those there their its it be has have had any all my our can you please find calculate get list than then per by at as been
when where why also both some i we us them""".split())
KEYWORDS = set("""average avg mean median total sum max min maximum minimum highest lowest most least top bottom count number numbers
monthly daily weekly yearly quarterly trend time over chart plot graph rate percentage percent ratio distribution value values record
records row rows data dataset common frequent popular best worst largest smallest generated sell sold many much unique different
column columns relation relationship correlation correlate between versus overall total there more less""".split())
SYN = {"revenue": ["sale"], "sale": ["revenue"], "sell": ["quantity", "qty"], "sold": ["quantity", "qty"],
       "product": ["item", "description"]}


def _words(q):
    return re.findall(r"[A-Za-z]{3,}", q.lower())


def match_columns(question, columns):
    """Columns referenced by the question, ordered by position in the question."""
    toks = [stem(t) for t in split_tokens(question)]
    pos = {}
    for i, t in enumerate(toks):
        pos.setdefault(t, i)
    ext = dict(pos)
    for t, i in list(pos.items()):
        for s in SYN.get(t, []):
            ext.setdefault(s, i + 0.5)
    cands = []
    operation_words = {
    "average", "avg", "mean", "median", "total", "sum",
    "max", "maximum", "min", "minimum", "highest", "lowest",
    "most", "least", "top", "bottom", "largest", "smallest",
}

    for c in columns:
        ct = [
            t for t in name_tokens(c)
            if t not in STOP and t not in operation_words
        ]
        hit = {t for t in ct if t in ext}
        if not ct or not hit or len(hit) / len(ct) < 0.5:
            continue
        cands.append((c, len(hit) / len(ct), hit, min(ext[t] for t in hit), sum(t in pos for t in hit)))
    keep = []
    for c in cands:
        dominated = any(o is not c and ((c[2] < o[2] and o[1] >= c[1]) or (c[2] == o[2] and (o[1], o[4]) > (c[1], c[4])))
                        for o in cands)
        if not dominated:
            keep.append(c)
    return [k[0] for k in sorted(keep, key=lambda x: x[3])]


def guess_subject(question, matched):
    known = set()
    for c in matched:
        known |= set(name_tokens(c))
    for w in _words(question):
        if w in STOP or w in KEYWORDS or stem(w) in known or stem(w) in KEYWORDS:
            continue
        return w
    return None


def detect_filters(question, df, profile):
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", question.lower()) + " "
    out = []
    for c, info in profile["columns"].items():
        if info["kind"] not in ("categorical", "boolean") or info["unique"] > 300:
            continue
        for v in df[c].dropna().unique():
            sv = re.sub(r"[^a-z0-9 ]", " ", str(v).lower()).strip()
            if sv.replace(" ", "").isdigit():
                continue
            short_upper = len(str(v)) == 2 and str(v).isupper() and re.search(rf"\b{re.escape(str(v))}\b", question)
            if (len(sv) >= 3 and f" {sv} " in ql) or short_upper:
                out.append((c, v))
                break
    return out


def detect_agg(ql):
    if re.search(r"\b(average|avg|mean)\b", ql): return "mean"
    if re.search(r"\bmedian\b", ql): return "median"
    if re.search(r"\b(std|standard deviation)\b", ql): return "std"
    if re.search(r"\b(total|sum|revenue|generated|sell|sold|sells|sales)\b", ql): return "sum"
    if re.search(r"\b(maximum|max)\b", ql): return "max"
    if re.search(r"\b(minimum|min)\b", ql): return "min"
    return None


def detect_rank(ql):
    if re.search(r"\b(lowest|least|fewest|smallest|worst|bottom|minimum)\b", ql): return "asc"
    if re.search(r"\b(highest|most|top|largest|biggest|best|greatest|maximum)\b", ql): return "desc"
    return None


def detect_freq(ql):
    for key, pat in (("day", r"daily|per day|by day|each day"), ("week", r"weekly|per week|by week|each week"),
                     ("month", r"monthly|per month|by month|each month"), ("quarter", r"quarterly|per quarter|by quarter"),
                     ("year", r"yearly|annual|annually|per year|by year|each year")):
        if re.search(rf"\b({pat})\b", ql):
            return key
    return None



def top_n(ql):
    patterns = [
        r"\b(?:top|bottom)\s+(\d+)\b",
        r"\b(\d+)\s+(?:most|top|highest|lowest|best|worst)\b",
        r"\b(?:which|show|list|find|get)?\s*(\d+)\s+\w+\s+(?:have|has|with)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, ql)
        if match:
            return int(match.group(1))

    return None



def detect_chart_type(question):
    """Detect an explicitly requested chart without changing query intent."""
    q = question.lower()

    patterns = [
        ("heatmap", r"\b(heat\s*map|correlation matrix)\b"),
        ("box", r"\b(box\s*plot|boxplot)\b"),
        ("scatter", r"\b(scatter\s*plot|scatterplot)\b"),
        ("hist", r"\b(histogram|histograms)\b"),
        ("pie", r"\b(pie\s*chart|piechart)\b"),
        ("line", r"\b(line\s*chart|line\s*graph)\b"),
        ("stacked_bar", r"\bstacked\s+bar\s+chart\b"),
        ("grouped_bar", r"\b(grouped|clustered)\s+bar\s+chart\b"),
        ("bar", r"\bbar\s+chart\b"),
    ]

    for chart_type, pattern in patterns:
        if re.search(pattern, q):
            return chart_type

    return None



def rule_plan(q, df, profile, ctx=None):
    ql = q.lower().strip()
    cols = profile["columns"]
    K = lambda c: cols[c]["kind"]
    filters = detect_filters(q, df, profile)
    fcols = {c for c, _ in filters}
    matched = match_columns(q, list(cols))
    mnum = [c for c in matched if K(c) == "numerical"]
    mcat = [c for c in matched if K(c) in ("categorical", "boolean") and c not in fcols]
    mid = [c for c in matched if K(c) == "identifier"]
    mdt = [c for c in matched if K(c) == "datetime"]
    mtxt = [c for c in matched if K(c) == "text"]
    has = lambda pat: re.search(pat, ql) is not None
    agg, rank, n = detect_agg(ql), detect_rank(ql), top_n(ql)
    subj = guess_subject(q, matched)
    chart_type = detect_chart_type(ql)

    print("RANK DEBUG:", {
    "matched": matched,
    "numeric": mnum,
    "ids": mid,
    "n": n,
    "rank": rank
})

    def P(intent, **kw):
        plan = {
            "intent": intent,
            "filters": filters,
            "n": n,
            "agg": agg,
            "rank": rank,
            **kw,
        }

        if chart_type:
            plan["chart_type"] = chart_type

        return plan

    if has(r"\b(why|explain|reason)\b") and ctx:
        return P("explain")
    if has(r"\bduplicat"): return P("duplicates")
    if has(r"\b(missing|null|nulls|nan|blank|empty)\b"): return P("missing", columns=matched)
    if has(r"\boutlier|\banomal|\bextreme values"): return P("outliers", column=mnum[0] if mnum else None)
    if has(r"\bskew"): return P("skewness", column=mnum[0] if mnum else None)
    if has(r"\bchurn"):
        cc = [c for c in cols if "churn" in name_tokens(c)]
        if has(r"\b(likely|predict|probab|will|risk)\b") and not has(r"\brate\b"): return P("churn_predict")
        return P("churn_rate", column=cc[0]) if cc else P("column_not_found", term="churn")
    if has(r"\b(how many|number of|count of|total number of)\s+(rows|records|entries|observations|lines)"): return P("row_count")
    
    if has(
        r"\b(how many|number of)\s+columns\b"
        r"|\bcolumn names\b"
        r"|\blist (all )?(the )?columns\b"
        r"|\bwhat columns\b"
        r"|\bwhich columns (are|exist)\b"
        r"|\b(list|show|give|display)\b.{0,30}\b(numeric|numerical)\s+columns\b"
        r"|\b(numeric|numerical)\s+columns\b"
    ):
        if has(r"\b(numeric|numerical)\s+columns\b"):
            numeric_cols = [
                c for c in profile["numerical"]
                if c in df.columns
            ]
            table = pd.DataFrame({
                "Column": numeric_cols,
                "Type": ["numerical"] * len(numeric_cols),
            })
            return P("columns")
        return P("columns")

    if has(r"\b(overview|summary|summarize|summarise|profile)\b|describe the (data|dataset)|about (the|this) (data|dataset)|dataset info") \
            and not (mnum or mcat or mid):
        return P("overview")

    rel = has(r"correlat|\brelat|\bvs\b|versus|impact|affect|influence|depend|associat|heatmap")
    if has(r"correlat|heatmap") and len(mnum) < 2:
        if has(r"\bbetween\b|\bvs\b|versus") and subj:
            return P("column_not_found", term=subj)
        return P("correlation", columns=[])
    if rel and len(mnum) >= 2: return P("correlation", columns=mnum[:2])
    if rel and mnum and mcat:
        return P("groupby", group_col=mcat[0], value_col=mnum[0], agg="mean", relation=True, rank=None)

    fq = detect_freq(ql)

    explicit_time_series = has(
    r"\bover time\b|\btrend\b|\btime series\b"
    )

    if (fq and profile["datetime"]) or explicit_time_series:
        dtc = mdt[0] if mdt else (profile["datetime"][0] if profile["datetime"] else None)
        if dtc is None: return P("column_not_found", term="date/time")
        val = mnum[0] if mnum else None
        if val is None and subj: return P("column_not_found", term=subj)
        return P("timeseries", date_col=dtc, value_col=val, freq=fq, agg=(agg or ("sum" if val else "count")))

    if has(r"\bwords?\b") and (mtxt or mcat):
        return P("text_words", column=(mtxt or mcat)[0])
    if has(r"\b(common|commonly|frequent|frequently|popular|complaints?)\b") and not mnum:
        if mtxt: return P("text_words", column=mtxt[0])
        if mcat: return P("value_counts", column=mcat[0], rank=rank or "desc")
        return P("column_not_found", term=subj or "that")

    by_word = has(r"\b(by|per|each|across)\b")
    count_like = has(r"\b(how many|number of|count)\b")
    if mcat and (mnum or mid or agg or rank or by_word):
        g = mcat[0]
        if mnum:
            a = agg if agg in ("mean", "median", "sum", "min", "max", "std") else "mean"
            return P("groupby", group_col=g, value_col=mnum[0], agg=a)
        if mid:
            return P("groupby", group_col=g, entity_col=mid[0], agg="nunique")
        if count_like and not subj:
            return P("groupby", group_col=g, agg="count")
        if subj:
            return P("column_not_found", term=subj)
        lp = ctx or {}
        if lp.get("intent") == "groupby" and lp.get("agg") and lp.get("agg") != "count":
            return P("groupby", group_col=g, value_col=lp.get("value_col"), entity_col=lp.get("entity_col"),
                     agg=lp["agg"], inherited=True)
        if lp.get("intent") == "stat" and lp.get("column"):
            return P("groupby", group_col=g, value_col=lp["column"], agg=lp.get("agg") or "mean", inherited=True)
        return P("clarify", group_col=g)
    if not matched and ctx and ctx.get("intent") == "groupby" and rank and not ctx.get("relation"):
        return P("groupby", group_col=ctx["group_col"], value_col=ctx.get("value_col"), entity_col=ctx.get("entity_col"),
                 agg=ctx.get("agg"), inherited=True)

    if count_like:
        c = (mid or mcat or [None])[0]
        return P("count_unique", column=c) if c else P("column_not_found", term=subj or "that")
    
    
    
    if mnum:
        # Handle top/bottom N record requests.
        if n is not None and rank in ("asc", "desc"):
            id_cols = [
                c for c in mid
                if c in cols and c in df.columns
            ]

            # Fallback to identifier columns in the dataset profile.
            if not id_cols:
                id_cols = [
                    c for c, info in cols.items()
                    if info.get("kind") == "identifier"
                    and c in df.columns
                ]

            # If there is no identifier, use the row index as a fallback.
            entity_col = id_cols[0] if id_cols else None

            return P(
                "top_n",
                column=mnum[0],
                entity_col=entity_col,
                n=n,
                rank=rank,
            )

        # Preserve existing single-value statistic behavior.
        if agg or rank:
            return P(
                "stat",
                column=mnum[0],
                agg=agg or ("max" if rank == "desc" else "min"),
            )

        return P("describe_column", column=mnum[0])



    if mcat: return P("value_counts", column=mcat[0])
    if mdt: return P("describe_column", column=mdt[0])
    if mtxt: return P("text_words", column=mtxt[0])
    if subj and (agg or rank or has(r"\b(average|total|count|rate|trend|how many|number of)\b")):
        return P("column_not_found", term=subj)
    
    # Handle explicit chart requests when no analysis intent matched.
    if chart_type:
        if mcat:
            return P("value_counts", column=mcat[0])
        if mnum:
            return P("describe_column", column=mnum[0])
        if mdt:
            return P("describe_column", column=mdt[0])
        return P("unrelated")

    return P("unrelated")



REQUIRED = {"stat": ["column"], "describe_column": ["column"], "groupby": ["group_col"], "count_unique": ["column"],
            "value_counts": ["column"], "text_words": ["column"], "timeseries": ["date_col"], "churn_rate": ["column"]}


def validate_plan(c, df, profile):
    """Accept an LLM-produced plan only if every field is valid for this dataset."""
    if not isinstance(c, dict) or c.get("intent") not in ALLOWED_INTENTS:
        return None
    cols = profile["columns"]
    plan = {"intent": c["intent"], "filters": [], "n": c.get("n") if isinstance(c.get("n"), int) else None,
            "agg": None, "rank": None}
    for k in ("column", "group_col", "value_col", "entity_col", "date_col"):
        v = c.get(k)
        if v in (None, ""):
            continue
        if v not in cols:
            return None
        plan[k] = v
    cl = c.get("columns") or []
    if any(x not in cols for x in cl):
        return None
    plan["columns"] = cl
    if c.get("agg") is not None:
        if c["agg"] not in ALLOWED_AGGS: return None
        plan["agg"] = c["agg"]
    if c.get("rank") in ("asc", "desc"): plan["rank"] = c["rank"]
    if c.get("freq") in FREQ: plan["freq"] = c["freq"]
    plan["relation"] = bool(c.get("relation"))
    for f in c.get("filters") or []:
        col, val = f.get("col"), f.get("value")
        if col not in cols or not (df[col].astype(str).str.lower() == str(val).lower()).any():
            return None
        plan["filters"].append((col, val))
    if any(k not in plan for k in REQUIRED.get(plan["intent"], [])):
        return None
    if plan["intent"] == "groupby" and plan["agg"] not in ("count", "nunique") and "value_col" not in plan:
        return None
    if plan["intent"] == "groupby" and plan["agg"] == "nunique" and "entity_col" not in plan:
        return None
    plan["term"] = str(c.get("missing_term") or "that")
    return plan


def _extract_json(text):
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])


def parse_question(question, df, profile, ctx_plan=None, snippets=None, llm_cfg=None):
    """Return a validated plan. Falls back to the rule-based plan if the LLM is unavailable/invalid."""
    base = rule_plan(question, df, profile, ctx_plan)
    base["source"] = "rules"
    if llm_cfg and llm_cfg.get("api_key") and client.available():
        try:
            raw = client.complete(SYSTEM_INTENT, build_intent_prompt(question, profile, snippets or [], ctx_plan),
                                  llm_cfg["api_key"], llm_cfg.get("model"))
            plan = validate_plan(_extract_json(raw), df, profile)
            if plan:
                # Preserve explicit chart requests from the rule-based parser.
                if base.get("chart_type"):
                    plan["chart_type"] = base["chart_type"]

                plan["source"] = "llm"
                return plan
        except Exception:
            pass
    return base
