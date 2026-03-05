import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(
    page_title="Alberta Bike Swap — Pricing Reference",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #999; }
    div[data-testid="stDataFrame"] { border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
    .price-pill {
        display: inline-block;
        background: #1a3a2a;
        color: #4ade80;
        border-radius: 20px;
        padding: 2px 12px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .range-pill {
        display: inline-block;
        background: #1a2a3a;
        color: #93c5fd;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.9rem;
    }
    .note-box {
        background: #1e1e2e;
        border-left: 3px solid #7c3aed;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #ccc;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Non-bike keywords ─────────────────────────────────────────────────────────
NON_BIKE_KEYWORDS = [
    "shirt","t-shirt","tshirt","tee","hoodie","jersey","jacket",
    "helmet","glove","gloves","shorts","pants","shoe","shoes",
    "bottle","bag","lock","pump","light","bell","grip","grips",
    "tube","tire","tyre","chain","pedal","saddle","seatpost",
    "handlebar","stem","cassette","derailleur","brake","cable",
    "tool","accessory","accessories","water","cap","hat","basket",
]

EXPECTED_COLS = {
    "manufacturer": ["manufacturer","brand","make"],
    "model":        ["model","bike","type"],
    "price":        ["price","amount","value","sale"],
    "status":       ["status","result","outcome"],
    "year":         ["year"],
    "location":     ["location","city","event","site"],
}

def detect_col(df_cols, candidates):
    lower = [c.lower() for c in df_cols]
    for cand in candidates:
        for i, col in enumerate(lower):
            if cand in col:
                return df_cols[i]
    return None

def parse_price(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,\s]", "", regex=True).replace("", np.nan),
        errors="coerce"
    )

def load_df(raw_bytes, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(io.BytesIO(raw_bytes))
    return pd.read_excel(io.BytesIO(raw_bytes))

def standardize(df, col_map, year_val, location_val):
    df = df.copy()
    rename = {}
    for std, src in col_map.items():
        if src and src in df.columns and src != std:
            rename[src] = std
    df = df.rename(columns=rename)

    if "year" not in df.columns or df["year"].isna().all():
        df["year"] = str(year_val) if year_val else "Unknown"
    if "location" not in df.columns or df["location"].isna().all():
        df["location"] = str(location_val) if location_val else "Unknown"

    if "price" in df.columns:
        df["price"] = parse_price(df["price"])

    for col in ["manufacturer", "model", "location"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().str.title()

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.upper()

    return df

def remove_non_bikes(df):
    mask = pd.Series(False, index=df.index)
    for col in ["manufacturer", "model"]:
        if col in df.columns:
            lower = df[col].astype(str).str.lower()
            for kw in NON_BIKE_KEYWORDS:
                mask |= lower.str.contains(rf"\b{kw}\b", na=False)
    return df[~mask].copy(), int(mask.sum())

def build_pricing_table(df):
    """
    Avg price  = mean of SOLD bikes with price > 0
    High/Low   = max/min of ALL bikes brought with price > 0
                 (includes unsold — reflects what owners asked)
    Times sold = count of SOLD rows per make/model
    Times seen = total rows per make/model (brought to any event)
    """
    df = df.copy()
    has_price = df[df["price"] > 0].copy()
    sold       = df[(df["status"] == "SOLD") & (df["price"] > 0)].copy()

    grp = ["manufacturer", "model"]

    # Range from ALL bikes brought with a price
    range_agg = has_price.groupby(grp)["price"].agg(
        high_price="max",
        low_price="min",
    )

    # Avg + count from SOLD only
    sold_agg = sold.groupby(grp)["price"].agg(
        avg_sold_price="mean",
        times_sold="count",
    )

    # Total times seen (any status, any price)
    seen_agg = df.groupby(grp).size().rename("times_brought")

    # Year/location info
    if "year" in df.columns:
        year_agg = df.groupby(grp)["year"].agg(lambda x: ", ".join(sorted(x.astype(str).unique())))
        year_agg.name = "years"
    if "location" in df.columns:
        loc_agg = df.groupby(grp)["location"].agg(lambda x: ", ".join(sorted(x.astype(str).unique())))
        loc_agg.name = "locations"

    pricing = (
        range_agg
        .join(sold_agg, how="outer")
        .join(seen_agg, how="outer")
    )
    if "year" in df.columns:
        pricing = pricing.join(year_agg, how="left")
    if "location" in df.columns:
        pricing = pricing.join(loc_agg, how="left")

    pricing = pricing.reset_index()
    pricing["times_sold"]    = pricing["times_sold"].fillna(0).astype(int)
    pricing["times_brought"] = pricing["times_brought"].fillna(0).astype(int)

    return pricing

# ═══════════════════════════════════ SIDEBAR ══════════════════════════════════
with st.sidebar:
    st.title("🚲 Alberta Bike Swap")
    st.caption("Pricing Reference Tool")
    st.divider()
    st.markdown("### 📂 Load Data")
    st.caption("Upload one CSV/Excel per event. Tag each with its year & location.")

    uploads = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    file_meta = {}
    if uploads:
        st.markdown("**File settings:**")
        for f in uploads:
            with st.expander(f"📄 {f.name}", expanded=True):
                yr_opt = st.selectbox("Year", ["2024", "2025", "Other"], key=f"yr_{f.name}")
                if yr_opt == "Other":
                    yr_opt = st.text_input("Enter year", key=f"yr_txt_{f.name}")
                loc_opt = st.selectbox("Location", ["Calgary", "Edmonton", "Other"], key=f"loc_{f.name}")
                if loc_opt == "Other":
                    loc_opt = st.text_input("Enter location", key=f"loc_txt_{f.name}")
                file_meta[f.name] = {"year": yr_opt, "location": loc_opt}

# ═══════════════════════════════════ LOAD DATA ════════════════════════════════
all_dfs = []
load_errors = []

if uploads:
    for f in uploads:
        try:
            raw = f.read()
            df_raw = load_df(raw, f.name)
            meta = file_meta.get(f.name, {})
            col_map = {std: detect_col(list(df_raw.columns), cands) for std, cands in EXPECTED_COLS.items()}
            df_std = standardize(df_raw, col_map, meta.get("year"), meta.get("location"))
            all_dfs.append(df_std)
        except Exception as e:
            load_errors.append(f"{f.name}: {e}")

if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all, n_nonbike = remove_non_bikes(df_all)

    for err in load_errors:
        st.error(f"⚠️ {err}")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("### 🔍 Filter")

        years = sorted(df_all["year"].dropna().astype(str).unique())
        sel_years = st.multiselect("Year", years, default=years)

        locs = sorted(df_all["location"].dropna().astype(str).unique())
        sel_locs = st.multiselect("Location", locs, default=locs)

    # Apply location + year filter before computing pricing
    fdf = df_all.copy()
    if sel_years:
        fdf = fdf[fdf["year"].astype(str).isin(sel_years)]
    if sel_locs:
        fdf = fdf[fdf["location"].astype(str).isin(sel_locs)]

    # ── Build pricing table ───────────────────────────────────────────────────
    pricing = build_pricing_table(fdf)

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total_sold     = int(pricing["times_sold"].sum())
    total_brought  = int(pricing["times_brought"].sum())
    total_revenue  = (fdf[(fdf["status"]=="SOLD") & (fdf["price"]>0)]["price"].sum())
    overall_avg    = (fdf[(fdf["status"]=="SOLD") & (fdf["price"]>0)]["price"].mean())
    unique_makes   = pricing["manufacturer"].nunique()

    # ═══════════════════════════════════ MAIN ════════════════════════════════
    st.title("Alberta Bike Swap — Pricing Reference")
    st.markdown(
        '<div class="note-box">'
        "The <strong>average price</strong> is calculated from bikes that actually sold. "
        "The <strong>price range</strong> (high / low) reflects all bikes brought to events with a listed price — "
        "including unsold bikes — showing the full spread owners have priced at."
        "</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🚲 Unique Make + Models", f"{len(pricing):,}")
    k2.metric("🏷 Brands",              f"{unique_makes:,}")
    k3.metric("✅ Total Sold",           f"{total_sold:,}")
    k4.metric("💰 Total Revenue",        f"${total_revenue:,.0f}")
    k5.metric("📊 Overall Avg Sale",     f"${overall_avg:,.0f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📋 Pricing Table", "🔎 Look Up a Bike", "📈 Insights"
    ])

    # ══════════════════════════ TAB 1: Full pricing table ═════════════════════
    with tab1:
        st.subheader("Pricing Reference Table")
        st.caption("All manufacturer + model combinations seen across selected events.")

        # Filter controls
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        with fc1:
            brand_search = st.text_input("Filter by brand", "")
        with fc2:
            model_search = st.text_input("Filter by model", "")
        with fc3:
            min_sold = st.number_input("Min. times sold", min_value=0, value=0, step=1)

        # Apply table filters
        tbl = pricing.copy()
        if brand_search:
            tbl = tbl[tbl["manufacturer"].str.contains(brand_search, case=False, na=False)]
        if model_search:
            tbl = tbl[tbl["model"].str.contains(model_search, case=False, na=False)]
        if min_sold > 0:
            tbl = tbl[tbl["times_sold"] >= min_sold]

        tbl = tbl.sort_values("times_sold", ascending=False).reset_index(drop=True)

        # Build display table
        display = pd.DataFrame()
        display["Brand"]          = tbl["manufacturer"]
        display["Model"]          = tbl["model"]
        display["Avg Sold Price"] = tbl["avg_sold_price"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
        display["Low Price"]  = tbl["low_price"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
        display["High Price"] = tbl["high_price"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
        display["Times Sold"]    = tbl["times_sold"]
        display["Times Brought"] = tbl["times_brought"]
        if "years" in tbl.columns:
            display["Year(s)"]   = tbl["years"]
        if "locations" in tbl.columns:
            display["Location(s)"] = tbl["locations"]

        st.dataframe(display, use_container_width=True, hide_index=True, height=550)
        st.caption(f"Showing {len(display):,} make + model combinations")

        st.download_button(
            "⬇️ Download pricing table as CSV",
            display.to_csv(index=False).encode(),
            "bike_swap_pricing_reference.csv",
            "text/csv",
        )

    # ══════════════════════════ TAB 2: Bike lookup ════════════════════════════
    with tab2:
        st.subheader("Look Up a Specific Bike")
        st.caption("Search by manufacturer and/or model to see its pricing history.")

        lc1, lc2 = st.columns(2)
        with lc1:
            brands_list = sorted(pricing["manufacturer"].dropna().unique())
            sel_brand = st.selectbox("Manufacturer / Brand", ["— Any —"] + brands_list)
        with lc2:
            if sel_brand != "— Any —":
                models_list = sorted(
                    pricing[pricing["manufacturer"] == sel_brand]["model"].dropna().unique()
                )
            else:
                models_list = sorted(pricing["model"].dropna().unique())
            sel_model = st.selectbox("Model", ["— Any —"] + models_list)

        result = pricing.copy()
        if sel_brand != "— Any —":
            result = result[result["manufacturer"] == sel_brand]
        if sel_model != "— Any —":
            result = result[result["model"] == sel_model]

        if len(result) == 0:
            st.info("No matching bikes found.")
        elif len(result) == 1:
            row = result.iloc[0]
            st.markdown(f"## {row['manufacturer']} — {row['model']}")

            m1, m2, m3, m4 = st.columns(4)
            avg_val  = f"${row['avg_sold_price']:,.0f}" if pd.notna(row.get("avg_sold_price")) else "No sales"
            low_val  = f"${row['low_price']:,.0f}"  if pd.notna(row.get("low_price"))  else "—"
            high_val = f"${row['high_price']:,.0f}" if pd.notna(row.get("high_price")) else "—"

            m1.metric("💰 Avg Sold Price",  avg_val)
            m2.metric("📉 Lowest Price",    low_val)
            m3.metric("📈 Highest Price",   high_val)
            m4.metric("✅ Times Sold",      f"{int(row['times_sold'])}")

            extra = []
            if pd.notna(row.get("years")):
                extra.append(f"**Years:** {row['years']}")
            if pd.notna(row.get("locations")):
                extra.append(f"**Locations:** {row['locations']}")
            extra.append(f"**Times brought to events:** {int(row['times_brought'])}")
            st.markdown("  ·  ".join(extra))

            # Show all individual sales for this bike
            st.markdown("#### All transactions for this bike")
            bike_rows = fdf[
                (fdf["manufacturer"].str.lower() == row["manufacturer"].lower()) &
                (fdf["model"].str.lower() == row["model"].lower())
            ][["manufacturer","model","price","status","year","location"]].copy()
            bike_rows = bike_rows.sort_values("price", ascending=False)
            bike_rows["price"] = bike_rows["price"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "—"
            )
            st.dataframe(bike_rows, use_container_width=True, hide_index=True)

        else:
            # Multiple results — show summary table
            st.markdown(f"**{len(result)} bike(s) match your selection:**")
            disp2 = pd.DataFrame({
                "Brand":          result["manufacturer"],
                "Model":          result["model"],
                "Avg Sold Price": result["avg_sold_price"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—"),
                "Low":            result["low_price"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—"),
                "High":           result["high_price"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—"),
                "Times Sold":     result["times_sold"],
            })
            st.dataframe(disp2.sort_values("Times Sold", ascending=False),
                         use_container_width=True, hide_index=True)

    # ══════════════════════════ TAB 3: Insights ═══════════════════════════════
    with tab3:
        st.subheader("Event Insights")

        ic1, ic2 = st.columns(2)

        with ic1:
            st.markdown("#### Top 15 Brands by Revenue")
            brand_rev = (
                fdf[(fdf["status"]=="SOLD") & (fdf["price"]>0)]
                .groupby("manufacturer")["price"].sum()
                .nlargest(15).reset_index()
            )
            brand_rev.columns = ["Brand","Revenue"]
            st.bar_chart(brand_rev.set_index("Brand"))

        with ic2:
            st.markdown("#### Top 15 Brands by # Bikes Sold")
            brand_ct = (
                fdf[fdf["status"]=="SOLD"]
                .groupby("manufacturer").size()
                .nlargest(15).reset_index(name="Sold")
            )
            st.bar_chart(brand_ct.set_index("manufacturer"))

        ic3, ic4 = st.columns(2)

        with ic3:
            st.markdown("#### Sale Price Distribution")
            sold_prices = fdf[(fdf["status"]=="SOLD") & (fdf["price"]>0)]["price"]
            if len(sold_prices):
                bins   = [0,50,100,150,200,300,400,500,750,1000,1500,2500]
                labels = [f"${bins[i]}–${bins[i+1]}" for i in range(len(bins)-1)]
                bc = pd.cut(sold_prices, bins=bins, labels=labels, right=False).value_counts().sort_index()
                st.bar_chart(bc)

        with ic4:
            st.markdown("#### Status Breakdown")
            if "status" in fdf.columns:
                sc = fdf["status"].value_counts().reset_index()
                sc.columns = ["Status","Count"]
                st.bar_chart(sc.set_index("Status"))

        # Most valuable models
        st.markdown("#### 🏆 Top 10 Highest Average Sale Price (min. 2 sold)")
        top_val = (
            pricing[pricing["times_sold"] >= 2]
            .nlargest(10, "avg_sold_price")
            [["manufacturer","model","avg_sold_price","high_price","low_price","times_sold"]]
            .copy()
        )
        top_val["avg_sold_price"] = top_val["avg_sold_price"].map("${:,.0f}".format)
        top_val["high_price"]     = top_val["high_price"].map("${:,.0f}".format)
        top_val["low_price"]      = top_val["low_price"].map("${:,.0f}".format)
        top_val.columns = ["Brand","Model","Avg Sold","High","Low","Times Sold"]
        st.dataframe(top_val, use_container_width=True, hide_index=True)

        # Year/location comparison if multiple loaded
        if len(locs) > 1 or len(years) > 1:
            st.markdown("#### Revenue by Year & Location")
            rev_yl = (
                fdf[(fdf["status"]=="SOLD") & (fdf["price"]>0)]
                .groupby(["year","location"])["price"].sum()
                .reset_index()
                .pivot(index="year", columns="location", values="price")
                .fillna(0)
            )
            st.bar_chart(rev_yl)

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.title("🚲 Alberta Bike Swap — Pricing Reference")
    st.info("👈 Upload your CSV or Excel file(s) in the sidebar to get started.")
    st.markdown("""
    ### What this tool does

    Given the bikes brought to Alberta Bike Swap events in **Calgary** and **Edmonton** over **2024 and 2025**,
    this dashboard answers:

    > *"For a Trek Marlin / Specialized Sirrus / Norco Storm — what does it typically sell for,
    and what's the price range?"*

    | Column | Meaning |
    |--------|---------|
    | **Avg Sold Price** | Mean price of all sold units of that make + model |
    | **Low Price** | Lowest price any unit of that bike was brought in at |
    | **High Price** | Highest price any unit of that bike was brought in at |
    | **Times Sold** | How many units sold across all selected events |
    | **Times Brought** | How many units total came to events (sold + unsold) |

    ### How to load data
    Upload one CSV per event sheet (Calgary 2024, Calgary 2025, Edmonton 2024, Edmonton 2025).
    Tag each file with its year and location in the sidebar.
    Required columns: `Manufacturer`, `Model`, `Price`, `Status`
    """)
