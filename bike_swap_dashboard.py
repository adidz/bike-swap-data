import streamlit as st
import pandas as pd
import numpy as np
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alberta Bike Swap Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #888; }
    div[data-testid="stDataFrame"] { border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
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

    for col in ["manufacturer","model","location"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().str.title()

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.upper()

    return df

def remove_non_bikes(df):
    mask = pd.Series(False, index=df.index)
    for col in ["manufacturer","model"]:
        if col in df.columns:
            lower = df[col].astype(str).str.lower()
            for kw in NON_BIKE_KEYWORDS:
                mask |= lower.str.contains(rf"\b{kw}\b", na=False)
    return df[~mask].copy(), int(mask.sum())

def remove_outliers(df):
    if "price" not in df.columns:
        return df, 0
    sold_prices = df[(df["price"] > 0) & (df.get("status", pd.Series(dtype=str)) == "SOLD")]["price"]
    if len(sold_prices) < 4:
        return df, 0
    q1, q3 = sold_prices.quantile(0.25), sold_prices.quantile(0.75)
    hi = q3 + 3.5 * (q3 - q1)
    outlier_mask = df["price"] > hi
    return df[~outlier_mask].copy(), int(outlier_mask.sum())

# ═════════════════════════════ SIDEBAR ═══════════════════════════════════════
with st.sidebar:
    st.title("🚲 Alberta Bike Swap")
    st.caption("Analytics Dashboard")
    st.divider()
    st.markdown("### 📂 Load Data")
    st.caption("Upload one CSV/Excel per event. Set the year & location for each file.")

    uploads = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv","xlsx","xls"],
        accept_multiple_files=True,
    )

    file_meta = {}
    if uploads:
        st.markdown("**File settings:**")
        for f in uploads:
            with st.expander(f"📄 {f.name}", expanded=True):
                yr_opt = st.selectbox("Year", ["2024","2025","Other"], key=f"yr_{f.name}")
                if yr_opt == "Other":
                    yr_opt = st.text_input("Enter year", key=f"yr_txt_{f.name}")
                loc_opt = st.selectbox("Location", ["Calgary","Edmonton","Other"], key=f"loc_{f.name}")
                if loc_opt == "Other":
                    loc_opt = st.text_input("Enter location", key=f"loc_txt_{f.name}")
                file_meta[f.name] = {"year": yr_opt, "location": loc_opt}

# ═════════════════════════════ LOAD + COMBINE ════════════════════════════════
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
    df_all, n_nonbike  = remove_non_bikes(df_all)
    df_all, n_outliers = remove_outliers(df_all)

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("### 🔍 Filters")

        years = sorted(df_all["year"].dropna().astype(str).unique())
        sel_years = st.multiselect("Year", years, default=years)

        locs = sorted(df_all["location"].dropna().astype(str).unique())
        sel_locs = st.multiselect("Location", locs, default=locs)

        if "manufacturer" in df_all.columns:
            brands = sorted(df_all["manufacturer"].dropna().unique())
            sel_brands = st.multiselect("Brand / Manufacturer", brands, default=brands)
        else:
            sel_brands = []

        if "status" in df_all.columns:
            statuses = sorted(df_all["status"].dropna().unique())
            sel_status = st.multiselect("Status", statuses, default=statuses)
        else:
            sel_status = []

        if "price" in df_all.columns:
            pvals = df_all["price"].dropna()
            pmin, pmax = int(pvals.min()), int(pvals.max())
            sel_price = st.slider("Price Range ($)", pmin, pmax, (pmin, pmax))

    # ── Apply filters ─────────────────────────────────────────────────────────
    fdf = df_all.copy()
    if sel_years:        fdf = fdf[fdf["year"].astype(str).isin(sel_years)]
    if sel_locs:         fdf = fdf[fdf["location"].astype(str).isin(sel_locs)]
    if sel_brands and "manufacturer" in fdf.columns:
                         fdf = fdf[fdf["manufacturer"].isin(sel_brands)]
    if sel_status and "status" in fdf.columns:
                         fdf = fdf[fdf["status"].isin(sel_status)]
    if "price" in fdf.columns:
        fdf = fdf[(fdf["price"] >= sel_price[0]) & (fdf["price"] <= sel_price[1])]

    # ── Computed metrics ──────────────────────────────────────────────────────
    sold_df    = fdf[fdf["status"] == "SOLD"]    if "status" in fdf.columns else pd.DataFrame()
    ret_df     = fdf[fdf["status"] == "RETURNED"] if "status" in fdf.columns else pd.DataFrame()
    donated_df = fdf[fdf["status"] == "DONATED"]  if "status" in fdf.columns else pd.DataFrame()

    total_rev = sold_df["price"].sum()  if "price" in sold_df.columns else 0
    avg_price = sold_df["price"].mean() if "price" in sold_df.columns and len(sold_df) else 0
    n_sold    = len(sold_df)
    n_ret     = len(ret_df)
    n_don     = len(donated_df)

    # ═════════════════════════════ MAIN UI ═══════════════════════════════════
    st.title("Alberta Bike Swap — Analytics Dashboard")
    for err in load_errors:
        st.error(f"⚠️ {err}")
    if n_nonbike or n_outliers:
        st.caption(f"🧹 Auto-cleaned: **{n_nonbike}** non-bike items removed · **{n_outliers}** price outliers removed")

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("📦 Total Records",  f"{len(fdf):,}")
    k2.metric("💰 Total Revenue",  f"${total_rev:,.0f}")
    k3.metric("📊 Avg Sale Price", f"${avg_price:,.0f}")
    k4.metric("✅ Sold",           f"{n_sold:,}")
    k5.metric("↩️ Returned",       f"{n_ret:,}")
    k6.metric("🎁 Donated",        f"{n_don:,}")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🏷 By Brand", "🚲 By Model", "📋 Status Breakdown", "🗃 Raw Data"
    ])

    # ══ TAB 1: Overview ══════════════════════════════════════════════════════
    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Revenue by Year & Location")
            if all(c in fdf.columns for c in ["location","year","price"]) and len(sold_df):
                rev_pivot = (
                    sold_df.groupby(["year","location"])["price"].sum()
                    .reset_index()
                    .pivot(index="year", columns="location", values="price")
                    .fillna(0)
                )
                st.bar_chart(rev_pivot)
            else:
                st.info("Load multiple files (with different years/locations) to see comparison.")

        with c2:
            st.subheader("Status Distribution")
            if "status" in fdf.columns:
                sc = fdf["status"].value_counts().reset_index()
                sc.columns = ["Status","Count"]
                st.bar_chart(sc.set_index("Status"))

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Sale Price Distribution")
            if "price" in sold_df.columns and len(sold_df):
                bins   = [0,50,100,150,200,300,400,500,750,1000,1500,2500]
                labels = [f"${bins[i]}–${bins[i+1]}" for i in range(len(bins)-1)]
                tmp = sold_df.copy()
                tmp["bucket"] = pd.cut(tmp["price"], bins=bins, labels=labels, right=False)
                bc = tmp["bucket"].value_counts().sort_index().reset_index()
                bc.columns = ["Price Range","Count"]
                st.bar_chart(bc.set_index("Price Range"))
            else:
                st.info("No sold items in current selection.")

        with c4:
            st.subheader("Top 10 Brands by Revenue")
            if "manufacturer" in fdf.columns and "price" in sold_df.columns:
                top = sold_df.groupby("manufacturer")["price"].sum().nlargest(10).reset_index()
                top.columns = ["Brand","Revenue"]
                st.bar_chart(top.set_index("Brand"))

    # ══ TAB 2: By Brand ══════════════════════════════════════════════════════
    with tab2:
        if "manufacturer" not in fdf.columns:
            st.info("No manufacturer/brand column found.")
        else:
            st.subheader("Brand Summary")

            bagg = fdf.groupby("manufacturer").agg(
                Items     =("manufacturer","count"),
                Revenue   =("price","sum"),
                Avg_Price =("price","mean"),
            ).reset_index()
            bagg.columns = ["Brand","Total Items","Total Revenue","Avg Sale Price"]

            if "status" in fdf.columns:
                for sv, sl in [("SOLD","Sold"),("RETURNED","Returned"),("DONATED","Donated")]:
                    ct = fdf[fdf["status"]==sv].groupby("manufacturer").size().reset_index(name=sl)
                    bagg = bagg.merge(ct, left_on="Brand", right_on="manufacturer", how="left").drop(columns=["manufacturer"], errors="ignore")
                    bagg[sl] = bagg[sl].fillna(0).astype(int)

            bagg = bagg.sort_values("Total Revenue", ascending=False).reset_index(drop=True)

            disp = bagg.copy()
            disp["Total Revenue"]  = disp["Total Revenue"].map("${:,.0f}".format)
            disp["Avg Sale Price"] = disp["Avg Sale Price"].map("${:,.0f}".format)
            st.dataframe(disp, use_container_width=True, hide_index=True)

            st.subheader("Top 20 Brands — Item Count")
            top20 = bagg.head(20)[["Brand","Total Items"]].set_index("Brand")
            st.bar_chart(top20)

    # ══ TAB 3: By Model ══════════════════════════════════════════════════════
    with tab3:
        if "model" not in fdf.columns:
            st.info("No model column found.")
        else:
            group_by = st.radio("Group by", ["Model only","Brand + Model"], horizontal=True)
            gcols = ["manufacturer","model"] if group_by == "Brand + Model" and "manufacturer" in fdf.columns else ["model"]

            magg = fdf.groupby(gcols).agg(
                Items    =(gcols[-1],"count"),
                Revenue  =("price","sum"),
                Avg_Price=("price","mean"),
            ).reset_index()

            if "status" in fdf.columns:
                for sv, sl in [("SOLD","Sold"),("RETURNED","Returned"),("DONATED","Donated")]:
                    ct = fdf[fdf["status"]==sv].groupby(gcols).size().reset_index(name=sl)
                    magg = magg.merge(ct, on=gcols, how="left")
                    magg[sl] = magg[sl].fillna(0).astype(int)

            magg = magg.sort_values("Revenue", ascending=False).reset_index(drop=True)

            disp_m = magg.copy()
            disp_m["Revenue"]   = disp_m["Revenue"].map("${:,.0f}".format)
            disp_m["Avg_Price"] = disp_m["Avg_Price"].map("${:,.0f}".format)
            disp_m.columns = [c.replace("_"," ") for c in disp_m.columns]

            st.subheader(f"Model Summary — {len(magg):,} unique model(s)")
            st.dataframe(disp_m, use_container_width=True, hide_index=True)

            st.subheader("Top 15 Models by Count")
            top_m = fdf.groupby("model").size().nlargest(15).reset_index(name="Count")
            st.bar_chart(top_m.set_index("model"))

    # ══ TAB 4: Status Breakdown ═══════════════════════════════════════════════
    with tab4:
        if "status" not in fdf.columns:
            st.info("No status column found.")
        else:
            status_counts = fdf["status"].value_counts()
            cols_k = st.columns(len(status_counts))
            for i, (s, cnt) in enumerate(status_counts.items()):
                pct = cnt / len(fdf) * 100
                rev = fdf[fdf["status"]==s]["price"].sum() if "price" in fdf.columns else 0
                with cols_k[i]:
                    st.metric(s, f"{cnt:,}", f"{pct:.1f}% of total")
                    if rev > 0:
                        st.caption(f"Revenue: ${rev:,.0f}")

            st.divider()

            # Year + location pivot
            if all(c in fdf.columns for c in ["year","location"]):
                st.subheader("Sold Count — Year × Location")
                p = (
                    fdf[fdf["status"]=="SOLD"]
                    .groupby(["year","location"]).size()
                    .reset_index(name="Count")
                    .pivot(index="year", columns="location", values="Count")
                    .fillna(0)
                )
                st.bar_chart(p)

            # Return rate by brand
            if "manufacturer" in fdf.columns:
                st.subheader("Return Rate by Brand (min. 3 items)")
                bs = fdf.groupby(["manufacturer","status"]).size().unstack(fill_value=0).reset_index()
                bs["Total"] = bs.drop(columns="manufacturer").sum(axis=1)
                bs = bs[bs["Total"] >= 3]
                if "RETURNED" in bs.columns:
                    bs["Return Rate %"] = (bs["RETURNED"] / bs["Total"] * 100).round(1)
                    tbl = bs[["manufacturer","Total","SOLD" if "SOLD" in bs.columns else "Total","RETURNED","Return Rate %"]].copy()
                    tbl.columns = (["Brand","Total Items"] +
                                   (["Sold"] if "SOLD" in bs.columns else []) +
                                   ["Returned","Return Rate %"])
                    st.dataframe(
                        tbl.sort_values("Return Rate %", ascending=False),
                        use_container_width=True, hide_index=True
                    )

    # ══ TAB 5: Raw Data ═══════════════════════════════════════════════════════
    with tab5:
        search = st.text_input("🔎 Search across all columns", "")
        if search:
            mask = fdf.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            fdf_show = fdf[mask]
        else:
            fdf_show = fdf

        st.subheader(f"Showing {len(fdf_show):,} of {len(fdf):,} records")
        st.dataframe(fdf_show, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download as CSV",
            fdf_show.to_csv(index=False).encode(),
            "bike_swap_filtered.csv",
            "text/csv",
        )

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.title("🚲 Alberta Bike Swap — Analytics Dashboard")
    st.info("👈 Upload your CSV or Excel file(s) in the sidebar to get started.")

    st.markdown("""
    ### How to use this dashboard

    **Step 1 — Upload your data**
    Upload one file per event (e.g. *Calgary 2024.csv*, *Edmonton 2025.csv*).
    Required columns: `Manufacturer`, `Model`, `Price`, `Status`

    **Step 2 — Set year & location per file**
    Since your files contain Calgary/Edmonton data separately, tag each file in the sidebar.

    **Step 3 — Filter & explore**

    | Tab | What you'll see |
    |-----|----------------|
    | 📈 Overview | Revenue by year/location, price distribution, top brands |
    | 🏷 By Brand | Full brand table — revenue, sold, returned, donated |
    | 🚲 By Model | Model breakdown, group by Brand + Model |
    | 📋 Status | Sold/Returned/Donated rates, return rate by brand |
    | 🗃 Raw Data | Searchable & downloadable table |

    **Automatic data cleaning**
    - Removes non-bike accessories (helmets, shirts, tools, etc.)
    - Removes price outliers (3.5 × IQR method on sold items)
    """)
