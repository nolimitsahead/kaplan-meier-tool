import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import streamlit as st

st.title("🎯 Vẽ Kaplan-Meier (OS/PFS)")

uploaded_file = st.file_uploader("📂 Chọn file Excel (.xlsx)", type=["xlsx"])
analysis_type = st.selectbox("🔍 Loại phân tích", ["OS", "PFS"])
day_unit = st.radio("📅 Đơn vị thời gian trong cột Time", ["d", "m"], index=1)
cutoff_month = st.number_input("✂️ Mốc tháng cutoff", min_value=1, value=24)

if uploaded_file and st.button("📈 Vẽ biểu đồ"):
    data = pd.read_excel(uploaded_file)
    data["Group"] = data["Group"].str.strip()
    data["Event"] = data["Event"].astype(int)

    if day_unit == "d":
        data["Time"] = data["Time"] / 30.4375

    if cutoff_month:
        data.loc[data["Time"] > cutoff_month, "Time"] = cutoff_month
        data.loc[(data["Time"] == cutoff_month) & (data["Event"] == 1), "Event"] = 0

    y_label = (
        f"Tỉ lệ sống còn toàn bộ {analysis_type}"
        if analysis_type == "OS"
        else f"Tỉ lệ sống không bệnh tiến triển {analysis_type}"
    )

    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    median_dict = {}

    for group in data["Group"].unique():
        group_data = data[data["Group"] == group]
        kmf.fit(
            durations=group_data["Time"],
            event_observed=group_data["Event"],
            label=group,
        )
        kmf.plot_survival_function(ci_show=True)
        median = kmf.median_survival_time_
        median_dict[group] = (
            round(median, 2) if median and median <= cutoff_month else "Not reached"
        )

    plt.title(f"Kaplan-Meier ({analysis_type})")
    plt.xlabel("Thời gian (tháng)")
    plt.ylabel(f"{y_label} (%)")
    plt.axhline(0.5, color="gray", linestyle="--")
    plt.legend(title="Nhóm điều trị")

    # HR và p-value
    group1, group2 = data["Group"].unique()
    lr_test = logrank_test(
        data[data["Group"] == group1]["Time"],
        data[data["Group"] == group2]["Time"],
        event_observed_A=data[data["Group"] == group1]["Event"],
        event_observed_B=data[data["Group"] == group2]["Event"],
    )
    cph = CoxPHFitter()
    data["Group_code"] = pd.Categorical(data["Group"]).codes
    cph.fit(
        data[["Time", "Event", "Group_code"]], duration_col="Time", event_col="Event"
    )
    summary = cph.summary

    hr = summary.loc["Group_code", "exp(coef)"]
    ci_low = summary.loc["Group_code", "exp(coef) lower 95%"]
    ci_up = summary.loc["Group_code", "exp(coef) upper 95%"]
    p_val = lr_test.p_value

    # 🆕 Thêm text dưới biểu đồ
    text_str = (
        f"{group1} Median {analysis_type}: {median_dict[group1]} tháng\n"
        f"{group2} Median {analysis_type}: {median_dict[group2]} tháng\n"
        f"HR = {hr:.2f} (95% CI: {ci_low:.2f}–{ci_up:.2f})\n"
        f"P = {p_val:.3f}"
    )
    plt.gcf().text(
        0.1,
        -0.15,
        text_str,
        fontsize=10,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"),
    )

    plt.tight_layout()
    st.pyplot(plt)

    # Kết quả chi tiết (text riêng bên dưới)
    st.markdown(
        f"""
    ### 📊 Kết quả:
    - **{group1} Median {analysis_type}**: {median_dict[group1]} tháng  
    - **{group2} Median {analysis_type}**: {median_dict[group2]} tháng  
    - **HR** = {hr:.2f} (95% CI: {ci_low:.2f} – {ci_up:.2f})  
    - **P-value** = {p_val:.3f}
    """
    )

    # 🆕 Export dữ liệu cho SPSS
    st.markdown("### 📤 Export dữ liệu dùng cho SPSS")
    export_filename = f"KM_{analysis_type}_for_SPSS.xlsx"
    export_df = data[["Time", "Event", "Group"]]

    # Hiển thị bảng trước khi export
    st.dataframe(export_df)

    # Nút tải file
    @st.cache_data
    def convert_df_to_excel(df):
        from io import BytesIO

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="KM_data")
        return output.getvalue()

    excel_data = convert_df_to_excel(export_df)
    st.download_button(
        label="📥 Tải dữ liệu Excel cho SPSS",
        data=excel_data,
        file_name=export_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
