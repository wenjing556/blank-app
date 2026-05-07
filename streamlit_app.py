import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
from datetime import time

# 页面配置
st.set_page_config(
    page_title="MASLD风险评估工具",
    page_icon="🫀",
    layout="wide"
)

# ========== 1. 定义真实β系数（基于你的OR值）==========

# 联合暴露（参考组：joint_exposure2 = 3，即偏早型 + SJL<1）
JOINT_EXP_OR = {
    "3": 1.000,   # 参考组：偏早型 + SJL<1
    "1": 1.114,   # 早型 + SJL<1 (3-1)
    "2": 1.236,   # 早型 + SJL≥1 (3-2)
    "4": 1.133,   # 偏早型 + SJL≥1 (3-4)
    "5": 1.061,   # 偏晚型 + SJL<1 (3-5)
    "6": 1.225,   # 偏晚型 + SJL≥1 (3-6)
    "7": 1.173,   # 完全晚型 + SJL<1 (3-7)
    "8": 1.520,   # 完全晚型 + SJL≥1 (3-8)
}

# 计算β系数
BETA_JOINT = {k: math.log(v) for k, v in JOINT_EXP_OR.items()}

# 协变量β系数
BETA_SEX_MALE = math.log(0.639)      # 男 vs 女
BETA_ETHNIC_WHITE = math.log(0.816)  # 白人 vs 非白人
BETA_AGE = math.log(1.006)           # 每1岁
BETA_TDI = math.log(1.037)           # 每1分
BETA_SMOKING_CURRENT = math.log(0.823)  # 当前吸烟 vs 从不/既往
BETA_ALCOHOL_DAILY = math.log(0.578)    # 每天饮酒 vs 从不
BETA_EDU_HIGH = math.log(1.118)      # 大学及以上 vs 其他
BETA_PA_INSUFFICIENT = math.log(1.720)  # 体力活动不足 vs 充足
BETA_DIET = math.log(0.872)          # 健康饮食每分
BETA_SLEEP_SHORT = math.log(0.878)   # 短睡眠(<7h) vs ≥7h
BETA_SHIFT_WORK = math.log(0.917)    # 有夜班 vs 无
BETA_DIABETES = math.log(0.328)      # 有糖尿病 vs 无
BETA_HYPERTENSION = math.log(0.688)  # 有高血压 vs 无
BETA_AST = math.log(1.033)           # 每1 U/L
BETA_UA = math.log(1.011)            # 每1 μmol/L

# BMI的真实OR（修正：1/0.019 ≈ 52.6）
BETA_BMI_OVERWEIGHT = math.log(1/0.019)  # ≈3.96

# 截距（使健康人群基准风险约为5%）
INTERCEPT = -3.5

# 默认值
DEFAULT_AGE = 45
DEFAULT_TDI = 0
DEFAULT_DIET = 3
DEFAULT_AST = 25
DEFAULT_UA = 300

# ========== 2. 社会时差和睡眠倾向计算 ==========
def calculate_social_jetlag(work_sleep_time, work_wake_time, free_sleep_time, free_wake_time):
    """计算社会时差 = |周末睡眠中点 - 工作日睡眠中点|（小时）"""
    def get_midpoint(sleep_hour, wake_hour):
        if wake_hour < sleep_hour:
            wake_hour += 24
        return (sleep_hour + wake_hour) / 2
    
    work_mid = get_midpoint(work_sleep_time, work_wake_time)
    free_mid = get_midpoint(free_sleep_time, free_wake_time)
    return abs(free_mid - work_mid)

def time_to_hours(t):
    return t.hour + t.minute / 60

def get_chronotype_code(chronotype):
    """根据编码规则：1=早型，2=偏早型，3=偏晚型，4=完全晚型"""
    chrono_map = {
        "早型 (Definite Morning)": 1,
        "偏早型 (Rather Morning)": 2,
        "偏晚型 (Rather Evening)": 3,
        "完全晚型 (Definite Evening)": 4,
    }
    return chrono_map.get(chronotype, 2)

def get_sjl_category(sjl_hours):
    """社会时差分类：≥1为1，<1为0"""
    return 1 if sjl_hours >= 1 else 0

def get_joint_exposure_code(chronotype_code, sjl_category):
    """根据chronotype和SJL获取联合暴露编码"""
    mapping = {
        (1, 0): 1, (1, 1): 2,
        (2, 0): 3, (2, 1): 4,
        (3, 0): 5, (3, 1): 6,
        (4, 0): 7, (4, 1): 8,
    }
    return mapping.get((chronotype_code, sjl_category), 3)

# ========== 3. 辅助函数 ==========
def calculate_bmi(weight, height):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return bmi

def calculate_risk_probability(inputs):
    lp = INTERCEPT
    contributions = {}
    
    # 联合暴露
    joint_code = inputs['joint_exposure']
    beta_joint = BETA_JOINT.get(str(joint_code), 0)
    if beta_joint != 0:
        lp += beta_joint
        contributions[f"睡眠节律类型({joint_code})"] = beta_joint
    
    # 性别
    if inputs['sex'] == 1:
        lp += BETA_SEX_MALE
        contributions["性别(男)"] = BETA_SEX_MALE
    
    # 种族
    if inputs['ethnic_white'] == 1:
        lp += BETA_ETHNIC_WHITE
        contributions["种族(白人)"] = BETA_ETHNIC_WHITE
    
    # 年龄
    age_effect = BETA_AGE * (inputs['age'] - DEFAULT_AGE)
    if abs(age_effect) > 0.0001:
        lp += age_effect
        contributions["年龄"] = age_effect
    
    # TDI
    tdi_effect = BETA_TDI * (inputs['tdi'] - DEFAULT_TDI)
    if abs(tdi_effect) > 0.0001:
        lp += tdi_effect
        contributions["TDI指数"] = tdi_effect
    
    # 吸烟
    if inputs['smoking_current'] == 1:
        lp += BETA_SMOKING_CURRENT
        contributions["当前吸烟"] = BETA_SMOKING_CURRENT
    
    # 饮酒
    if inputs['alcohol_daily'] == 1:
        lp += BETA_ALCOHOL_DAILY
        contributions["每天饮酒"] = BETA_ALCOHOL_DAILY
    
    # 教育
    if inputs['education_high'] == 1:
        lp += BETA_EDU_HIGH
        contributions["高等教育"] = BETA_EDU_HIGH
    
    # 体力活动
    if inputs['physical_inactivity'] == 1:
        lp += BETA_PA_INSUFFICIENT
        contributions["体力活动不足"] = BETA_PA_INSUFFICIENT
    
    # 健康饮食
    diet_effect = BETA_DIET * (inputs['diet_score'] - DEFAULT_DIET)
    if abs(diet_effect) > 0.0001:
        lp += diet_effect
        contributions["健康饮食评分"] = diet_effect
    
    # 短睡眠
    if inputs['short_sleep'] == 1:
        lp += BETA_SLEEP_SHORT
        contributions["短睡眠(<7h)"] = BETA_SLEEP_SHORT
    
    # 夜班
    if inputs['shift_work'] == 1:
        lp += BETA_SHIFT_WORK
        contributions["夜班工作"] = BETA_SHIFT_WORK
    
    # 糖尿病
    if inputs['diabetes'] == 1:
        lp += BETA_DIABETES
        contributions["糖尿病"] = BETA_DIABETES
    
    # 高血压
    if inputs['hypertension'] == 1:
        lp += BETA_HYPERTENSION
        contributions["高血压"] = BETA_HYPERTENSION
    
    # BMI
    if inputs['bmi_overweight'] == 1:
        lp += BETA_BMI_OVERWEIGHT
        contributions["超重/肥胖(BMI≥25)"] = BETA_BMI_OVERWEIGHT
    
    # AST
    ast_effect = BETA_AST * (inputs['ast'] - DEFAULT_AST)
    if abs(ast_effect) > 0.0001:
        lp += ast_effect
        contributions["AST升高"] = ast_effect
    
    # 尿酸
    ua_effect = BETA_UA * (inputs['ua'] - DEFAULT_UA)
    if abs(ua_effect) > 0.001:
        lp += ua_effect
        contributions["尿酸升高"] = ua_effect
    
    probability = 1 / (1 + math.exp(-lp))
    return probability, contributions

def get_risk_level(probability):
    if probability < 0.05:
        return "极低风险", "🟢", "概率 < 5%，状态良好"
    elif probability < 0.10:
        return "低风险", "🟢", "概率 5-10%，基本健康"
    elif probability < 0.20:
        return "中低风险", "🟡", "概率 10-20%，建议改善生活方式"
    elif probability < 0.35:
        return "中风险", "🟠", "概率 20-35%，建议就医检查"
    else:
        return "高风险", "🔴", "概率 > 35%，强烈建议立即就医"

# ========== 4. 侧边栏输入 ==========
with st.sidebar:
    st.header("📋 填写以下信息")
    
    st.subheader("🌙 昼夜节律相关")
    
    # 睡眠倾向
    chronotype = st.selectbox(
        "睡眠倾向 (Chronotype)",
        ["早型 (Definite Morning)", "偏早型 (Rather Morning)", 
         "偏晚型 (Rather Evening)", "完全晚型 (Definite Evening)"]
    )
    
    st.markdown("**工作日作息**")
    work_sleep = st.time_input("工作日入睡时间", time(23, 0))
    work_wake = st.time_input("工作日起床时间", time(7, 0))
    
    st.markdown("**周末/休息日作息**")
    free_sleep = st.time_input("周末入睡时间", time(23, 30))
    free_wake = st.time_input("周末起床时间", time(8, 30))
    
    # 计算社会时差
    work_sleep_h = time_to_hours(work_sleep)
    work_wake_h = time_to_hours(work_wake)
    free_sleep_h = time_to_hours(free_sleep)
    free_wake_h = time_to_hours(free_wake)
    
    sjl_hours = calculate_social_jetlag(work_sleep_h, work_wake_h, free_sleep_h, free_wake_h)
    sjl_category = get_sjl_category(sjl_hours)
    
    st.caption(f"📊 社会时差：**{sjl_hours:.1f}小时** ({'≥1小时' if sjl_category else '<1小时'})")
    
    # 其他节律相关（修复：统一使用float类型）
    sleep_hours = st.number_input("平均睡眠时长(小时/天)", 0.0, 24.0, 7.0, step=0.5)
    shift_work = st.selectbox("有夜班工作史", ["无", "有"])
    
    st.subheader("👤 基本信息")
    sex = st.selectbox("性别", ["女", "男"])
    age = st.slider("年龄(岁)", 18, 80, 45)
    ethnic = st.selectbox("种族", ["非白人", "白人"])
    tdi = st.slider("TDI贫困指数", -5.0, 10.0, 0.0, 0.5)
    education = st.selectbox("教育水平", ["高中及以下", "大学及以上"])
    
    st.subheader("🏃 生活方式")
    smoking = st.selectbox("吸烟状态", ["从不/既往", "当前吸烟"])
    physical_activity = st.selectbox("体力活动", ["充足(≥150min/周)", "不足(<150min/周)"])
    diet_score = st.slider("健康饮食评分(0-5分)", 0, 5, 3)
    
    alcohol_freq = st.selectbox(
        "饮酒频率",
        ["从不/特殊场合", "每月1-3次", "每周1-2次", "每周3-4次", "每周5-6次", "每天或几乎每天"]
    )
    
    st.subheader("📏 身体测量")
    height = st.number_input("身高(cm)", 100.0, 220.0, 165.0, step=1.0)
    weight = st.number_input("体重(kg)", 30.0, 200.0, 65.0, step=1.0)
    
    st.subheader("⚕️ 健康状况")
    diabetes = st.selectbox("糖尿病", ["无", "有"])
    hypertension = st.selectbox("高血压", ["无", "有"])
    
    st.subheader("🧪 实验室检查")
    ast = st.number_input("AST (U/L)", 0.0, 200.0, 25.0, step=1.0)
    ua = st.number_input("尿酸 (μmol/L)", 0.0, 600.0, 300.0, step=1.0)

# ========== 5. 数据处理 ==========
# 计算BMI
bmi = calculate_bmi(weight, height)
bmi_overweight = 1 if bmi >= 25 else 0

# 获取编码
chronotype_code = get_chronotype_code(chronotype)
sjl_category = get_sjl_category(sjl_hours)
joint_exposure_code = get_joint_exposure_code(chronotype_code, sjl_category)

# 协变量编码
sex_code = 1 if sex == "男" else 0
ethnic_white_code = 1 if ethnic == "白人" else 0
smoking_current_code = 1 if smoking == "当前吸烟" else 0
alcohol_daily_code = 1 if alcohol_freq == "每天或几乎每天" else 0
education_high_code = 1 if education == "大学及以上" else 0
physical_inactivity_code = 1 if physical_activity == "不足(<150min/周)" else 0
short_sleep_code = 1 if sleep_hours < 7 else 0
shift_work_code = 1 if shift_work == "有" else 0
diabetes_code = 1 if diabetes == "有" else 0
hypertension_code = 1 if hypertension == "有" else 0

inputs = {
    'joint_exposure': joint_exposure_code,
    'sex': sex_code,
    'ethnic_white': ethnic_white_code,
    'age': age,
    'tdi': tdi,
    'smoking_current': smoking_current_code,
    'alcohol_daily': alcohol_daily_code,
    'education_high': education_high_code,
    'physical_inactivity': physical_inactivity_code,
    'diet_score': diet_score,
    'short_sleep': short_sleep_code,
    'shift_work': shift_work_code,
    'diabetes': diabetes_code,
    'hypertension': hypertension_code,
    'bmi_overweight': bmi_overweight,
    'ast': ast,
    'ua': ua
}

probability, contributions = calculate_risk_probability(inputs)
risk_level, risk_icon, risk_advice = get_risk_level(probability)

# 获取OR值显示
joint_or = JOINT_EXP_OR.get(str(joint_exposure_code), 1.0)

# ========== 6. 主界面展示 ==========
st.title("🫀 MASLD风险评估工具")
st.markdown("基于UK Biobank 30万人数据 | **社会时差 + 睡眠倾向** 联合暴露与MASLD关联分析")

# 第一行：关键信息
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(f"⏰ 社会时差：**{sjl_hours:.1f}小时**")
with col2:
    st.info(f"🌙 睡眠倾向：**{chronotype}**")
with col3:
    st.info(f"📏 BMI：**{bmi:.1f}** {'(≥25)' if bmi_overweight else '(<25)'}")
with col4:
    st.info(f"📊 联合暴露类型：**{joint_exposure_code}** (OR={joint_or:.3f})")

# 第二行：风险概率
col_main1, col_main2, col_main3 = st.columns([1, 2, 1])
with col_main2:
    if probability < 0.10:
        bg_color = '#d4edda'
        text_color = '#155724'
    elif probability < 0.20:
        bg_color = '#fff3cd'
        text_color = '#856404'
    elif probability < 0.35:
        bg_color = '#ffe5b4'
        text_color = '#cc7000'
    else:
        bg_color = '#f8d7da'
        text_color = '#721c24'
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: {bg_color}; border-radius: 10px;">
        <h2 style="margin: 0; color: {text_color};">{risk_icon} {risk_level}</h2>
        <h1 style="margin: 10px 0; font-size: 48px; color: {text_color};">{probability:.1%}</h1>
        <p style="margin: 0; color: {text_color};">{risk_advice}</p>
    </div>
    """, unsafe_allow_html=True)

# 第三行：详细分析
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 风险因素贡献")
    
    if contributions:
        contrib_df = pd.DataFrame(list(contributions.items()), columns=["风险因素", "贡献值"])
        contrib_df = contrib_df[abs(contrib_df["贡献值"]) > 0.001]
        contrib_df = contrib_df.sort_values("贡献值", ascending=False)
        
        fig = px.bar(contrib_df, x="贡献值", y="风险因素", orientation='h',
                     title="各因素对风险概率的贡献（正值=增加风险）",
                     color="贡献值", color_continuous_scale="Reds",
                     text="贡献值")
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=450, xaxis_title="贡献值 (β×X)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ 无显著风险因素，继续保持！")

with col_right:
    st.subheader("🔍 联合暴露解读")
    
    st.markdown(f"""
    **您的睡眠节律特征**：
    - 睡眠倾向：**{chronotype}** (编码: {chronotype_code})
    - 社会时差：**{sjl_hours:.1f}小时** ({'≥1' if sjl_category else '<1'})
    - 联合暴露类型：**{joint_exposure_code}**
    
    **相对于参考组（偏早型 + SJL<1）**：
    - OR = **{joint_or:.3f}**
    - 风险 {'升高' if joint_or > 1 else '降低'} {abs(joint_or-1)*100:.1f}%
    """)
    
    st.markdown("---")
    st.subheader("📐 核心证据")
    
    evidence_data = pd.DataFrame([
        ["超重/肥胖(BMI≥25)", "52.60", f"{BETA_BMI_OVERWEIGHT:.3f}"],
        ["体力活动不足", "1.720", f"{BETA_PA_INSUFFICIENT:.3f}"],
        ["高等教育", "1.118", f"{BETA_EDU_HIGH:.3f}"],
        ["AST升高(每U/L)", "1.033", f"{BETA_AST:.3f}"],
        ["尿酸升高(每μmol/L)", "1.011", f"{BETA_UA:.3f}"],
        ["年龄(每岁)", "1.006", f"{BETA_AGE:.3f}"],
        ["TDI指数", "1.037", f"{BETA_TDI:.3f}"],
        ["健康饮食(每分)", "0.872", f"{BETA_DIET:.3f}"],
        ["短睡眠(<7h)", "0.878", f"{BETA_SLEEP_SHORT:.3f}"],
    ], columns=["风险因素", "OR", "β系数"])
    
    st.dataframe(evidence_data, use_container_width=True, hide_index=True)
    
    st.caption("注：OR>1表示增加风险，OR<1表示降低风险")

# 第四行：个性化建议
st.markdown("---")
st.subheader("💡 个性化干预建议")

col_adv1, col_adv2 = st.columns(2)

with col_adv1:
    st.markdown("#### 🎯 针对您的风险水平")
    if probability >= 0.20:
        st.warning("""
        **立即行动**：
        - 预约肝脏B超检查
        - 检测肝功能(ALT/AST)
        - 咨询医生或营养师
        - 如超重，制定减重计划
        """)
    elif probability >= 0.10:
        st.info("""
        **近期改善**：
        - 3个月内复查肝功能
        - 开始规律运动和饮食调整
        - 记录饮食和睡眠日记
        - 减少酒精摄入
        """)
    else:
        st.success("""
        **保持与预防**：
        - 保持当前良好习惯
        - 每年体检关注肝功能
        - 预防体重增加
        - 维持规律作息
        """)

with col_adv2:
    st.markdown("#### 🌙 针对您的睡眠节律")
    
    if sjl_category == 1:
        st.warning("""
        **减少社会时差的方法**：
        - 周末起床时间不要比平时晚>1小时
        - 周末午间小憩(20-30分钟)替代睡懒觉
        - 周一早晨增加光照暴露
        - 逐步调整，每天改变15-30分钟
        """)
    else:
        st.success("✅ 您的社会时差控制良好，继续保持！")
    
    if chronotype_code in [3, 4]:
        st.info("""
        **针对晚型人的建议**：
        - 早晨增加光照暴露
        - 睡前1小时减少蓝光
        - 如条件允许，选择弹性工作时间
        """)

# 页脚
st.markdown("---")
st.caption("⚠️ 本工具基于UK Biobank、中山数据构建，仅供参考，不能替代专业医疗诊断。")
