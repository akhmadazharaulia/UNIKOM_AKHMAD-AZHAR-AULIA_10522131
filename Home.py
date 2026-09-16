import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sedjenak Mie & Kopi",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CSS hack: paksa st.toast() muncul di tengah layar
# (Streamlit belum punya parameter posisi resmi untuk toast,
# jadi ini nembak elemen DOM-nya langsung. Kalau update Streamlit
# di kemudian hari bikin ini geser lagi, tinggal sesuaikan selector-nya.)
# ==========================
st.markdown("""
<style>
div[data-testid="stToast"] {
    position: fixed !important;
    top: 5% !important;
    left: 50% !important;
    right: auto !important;
    bottom: auto !important;
    transform: translate(-50%, 0) !important;
    min-width: 320px;
    max-width: 480px;
    z-index: 99999 !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Session State
# ==========================

defaults = {
    "df_pos_raw": None,
    "df_pos_clean": None,
    "df_daily": None,
    "df_bom": None,
    "df_train": None,
    "df_test": None,
    "sarima_result": None,
    "forecast_df": None,
    "eval_df": None,
    "role": "Owner",
    "toast_msg": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================
# Helper: deteksi kolom penting secara otomatis
# ==========================

def _find_col(cols, *keywords):
    """Cari kolom pertama yang namanya mengandung salah satu keyword (case-insensitive)."""
    for c in cols:
        cl = str(c).lower()
        if any(k in cl for k in keywords):
            return c
    return None


def detect_pos_columns(df):
    cols = list(df.columns)
    return {
        "date": _find_col(cols, "date", "tanggal", "waktu", "receipt date"),
        "menu": _find_col(cols, "item", "menu", "produk", "nama"),
        "total": _find_col(cols, "total", "gross sales", "omzet", "subtotal", "amount", "harga"),
    }


def detect_bahan_column(df):
    cols = list(df.columns)
    col = _find_col(cols, "bahan", "ingredient", "material", "resep")
    return col if col else cols[0]


def format_rupiah(value):
    try:
        return f"Rp {value:,.0f}".replace(",", ".")
    except Exception:
        return "-"


def format_periode(dt):
    bulan_en = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{dt.year} {bulan_en[dt.month - 1]}"


def format_rupiah_short(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}M"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}jt"
    if value >= 1_000:
        return f"{value / 1_000:.0f}rb"
    return f"{value:.0f}"


# ==========================
# Sidebar
# ==========================

# ==========================
# Sidebar — dengan konfirmasi saat pindah mode
# ==========================

ROLE_OPTIONS = ["Owner", "Developer"]

if "pending_role" not in st.session_state:
    st.session_state["pending_role"] = None

st.sidebar.title("Mode Aplikasi")

radio_value = st.sidebar.radio(
    "Pilih Tampilan",
    ROLE_OPTIONS,
    index=ROLE_OPTIONS.index(st.session_state["role"]),
    key="role_radio"
)

# Kalau user memilih opsi yang beda dari mode aktif, munculkan popup konfirmasi
if radio_value != st.session_state["role"] and st.session_state["pending_role"] is None:
    st.session_state["pending_role"] = radio_value
    st.rerun()


@st.dialog("Konfirmasi Perpindahan Mode")
def confirm_mode_switch():
    target = st.session_state["pending_role"]
    st.write(f"Kamu akan pindah ke mode **{target}**.")
    st.caption("Tampilan dan menu yang aktif akan berubah mengikuti mode ini.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Ya, Pindah", use_container_width=True):
            st.session_state["role"] = target
            st.session_state["pending_role"] = None
            st.session_state["toast_msg"] = (f"Berhasil pindah ke mode {target}.", "✅")
            st.rerun()
    with c2:
        if st.button("❌ Batal", use_container_width=True):
            st.session_state["pending_role"] = None
            st.session_state["role_radio"] = st.session_state["role"]
            st.session_state["toast_msg"] = ("Perpindahan mode dibatalkan.", "❌")
            st.rerun()


if st.session_state["pending_role"] is not None:
    confirm_mode_switch()

# Tampilkan toast hasil aksi terakhir (kalau ada), lalu bersihkan agar tidak muncul berulang
if st.session_state.get("toast_msg"):
    _msg, _icon = st.session_state["toast_msg"]
    st.toast(_msg, icon=_icon)
    st.session_state["toast_msg"] = None

role = st.session_state["role"]

st.sidebar.divider()

if role == "Owner":
    st.sidebar.success("Owner")
else:
    st.sidebar.info("Developer")

# ==========================
# Header
# ==========================

if role == "Owner":

    st.title("Sedjenak Mie & Kopi")
    st.subheader("Sistem Prediksi Penjualan & Estimasi Kebutuhan Bahan Baku")

    st.markdown("""
    Aplikasi ini membantu pemilik usaha memperkirakan penjualan berdasarkan data transaksi
    dari **Loyverse POS** dan menghitung estimasi kebutuhan bahan baku sebagai dasar
    perencanaan operasional.
    """)

    st.divider()

else:

    st.title(" Sedjenak Mie & Kopi")
    st.subheader("Sistem Prediksi Penjualan Harian Menggunakan Metode SARIMA")

    st.markdown("""
    Aplikasi ini mengimplementasikan metode **Seasonal AutoRegressive Integrated Moving Average (SARIMA)**
    untuk melakukan prediksi penjualan harian berdasarkan data transaksi historis
    dari **Loyverse POS**, kemudian mengonversi hasil prediksi menjadi estimasi
    kebutuhan bahan baku menggunakan **Bill of Materials (BOM)**.
    """)

    st.divider()

    # ==========================
    # Feature
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Prediksi Penjualan**

        Menghasilkan estimasi penjualan
        untuk periode yang dipilih.
        """)

    with col2:
        st.info("""
        **Estimasi Bahan Baku**

        Menghitung kebutuhan bahan baku
        berdasarkan hasil prediksi.
        """)

    with col3:
        st.info("""
        **Evaluasi Model**

        Menampilkan MAE, RMSE,
        MAPE dan hasil evaluasi model.
        """)

    st.divider()

# ==========================
# Upload
# ==========================

st.header("Import Data")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Data Penjualan")

    st.caption("""
    Unggah data transaksi yang telah diekspor
    dari aplikasi Loyverse POS.
    """)

    pos_file = st.file_uploader(
        "Upload Data Penjualan",
        type=["csv", "xlsx", "xls"],
        key="pos_upload"
    )

    if pos_file is not None:

        try:
            if pos_file.name.endswith(".csv"):
                df_pos = pd.read_csv(pos_file)
            else:
                df_pos = pd.read_excel(pos_file)

            st.session_state["df_pos_raw"] = df_pos

            st.success("Data penjualan berhasil diunggah.")
            st.toast("Data penjualan berhasil diunggah!", icon="✅")

            with st.expander("Preview Data"):
                st.dataframe(df_pos.head())

        except Exception as e:
            st.error(f"Gagal membaca file data penjualan: {e}")
            st.toast("Gagal mengunggah data penjualan.", icon="❌")

with col2:

    st.markdown("### Data Bahan Baku")

    st.caption("""
    Unggah data komposisi bahan baku
    apabila terdapat perubahan menu atau resep.
    """)

    bom_file = st.file_uploader(
        "Upload Data BOM",
        type=["csv", "xlsx", "xls"],
        key="bom_upload"
    )

    if bom_file is not None:

        try:
            if bom_file.name.endswith(".csv"):
                df_bom = pd.read_csv(bom_file)
            else:
                df_bom = pd.read_excel(bom_file)

            st.session_state["df_bom"] = df_bom

            st.success("Data bahan baku berhasil diunggah.")
            st.toast("Data bahan baku berhasil diunggah!", icon="✅")

            with st.expander("Preview Data"):
                st.dataframe(df_bom.head())

        except Exception as e:
            st.error(f"Gagal membaca file data bahan baku: {e}")
            st.toast("Gagal mengunggah data bahan baku.", icon="❌")

st.divider()

# ==========================
# Status
# ==========================

st.header("Status Data")

if role == "Owner":

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Data Penjualan",
            "✅ Berhasil Diunggah" if st.session_state["df_pos_raw"] is not None else "⏳ Belum Diunggah"
        )

    with c2:
        st.metric(
            "Data Bahan Baku",
            "✅ Berhasil Diunggah" if st.session_state["df_bom"] is not None else "⏳ Belum Diunggah"
        )

    with c3:

        if (
            st.session_state["df_pos_raw"] is not None and
            st.session_state["df_bom"] is not None
        ):
            status = "✅ Siap Digunakan"
        else:
            status = "⏳ Load Data"

        st.metric(
            "Status Sistem",
            status
        )

else:

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Data POS",
            "Terupload ✓" if st.session_state["df_pos_raw"] is not None else "Belum Ada"
        )

    with c2:
        st.metric(
            "Data BOM",
            "Terupload ✓" if st.session_state["df_bom"] is not None else "Belum Ada"
        )

    with c3:
        st.metric(
            "Model SARIMA",
            "Terlatih ✓" if st.session_state["sarima_result"] is not None else "Belum Dilatih"
        )

    with c4:
        st.metric(
            "Hasil Prediksi",
            "Tersedia ✓" if st.session_state["forecast_df"] is not None else "Belum Ada"
        )