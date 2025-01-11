import streamlit as st
import pandas as pd
import numpy as np
import cost_of_living_in_Canada_object as cpo
from scipy import stats

st.set_page_config(layout="wide")

with st.container():
    st.write("# Opportunity Cost of living in Canada")
    exchange_rate = st.number_input("exchange rate CAD to HKD", min_value=0.0, max_value=1_000_000_000.0, value=5.7)
    # row 1
    col1_1, col1_2, col1_3 = st.columns(3)
    with col1_3:
        option = st.selectbox(
            "Do you want to show HKD or CAD?",
            ("HKD", "CAD"),
        )

    # row 2
    col2_1, col2_2, col2_3 = st.columns(3)
    Salary_HK = col2_1.number_input("Wage in HK in HKD annually", min_value=0.0, max_value=1_000_000_000.0, value=float(25000 * 12))
    Salary_CA = col2_2.number_input("Wage in Canada in CAD annually before tax", min_value=0.0, max_value=1_000_000_000.0, value=float(60000))
    Salary_CA_after_tax = cpo.income_before_to_income_after_tax_interpolation(Salary_CA)
    wage_difference_HKD = Salary_HK - Salary_CA_after_tax * exchange_rate  # Convert CAD to HKD before comparing
    wage_difference_CAD = wage_difference_HKD / exchange_rate
    col2_3.write(f"#### Wage Difference ({option}) after tax HK - CA:")
    if option == "HKD":
        col2_3.write(f"{wage_difference_HKD:,.2f} {option}")
    elif option == "CAD":
        col2_3.write(f"{wage_difference_CAD:,.2f} {option}")
    else:
        col2_3.write(f"ERROR")

    # row 3
    col3_1, col3_2, col3_3 = st.columns(3)
    R_HK_Growth = col3_1.number_input("Wage growth in HK percentage", min_value=0.0, max_value=100.0, value=3.0) / 100
    R_CA_Growth = col3_2.number_input("Wage growth in Canada percentage", min_value=0.0, max_value=100.0, value=3.0) / 100

    # row 4
    col4_1, col4_2, col4_3 = st.columns(3)
    Cost_living_HK = col4_1.number_input("Cost of living in HK in HKD annually", min_value=0.0, max_value=1_000_000_000.0, value=float(10000 * 12))
    Cost_living_CA = col4_2.number_input("Cost of living in Canada in CAD annually", min_value=0.0, max_value=1_000_000_000.0, value=float(3000 * 12))
    cost_of_living_difference_HKD = Cost_living_HK - Cost_living_CA * exchange_rate  # Convert CAD to HKD before comparing
    cost_of_living_difference_CAD = cost_of_living_difference_HKD / exchange_rate
    col4_3.write(f"#### Cost of Living Difference ({option}) HK - CA:")
    if option == "HKD":
        col4_3.write(f"{cost_of_living_difference_HKD:,.2f} {option}")
    elif option == "CAD":
        col4_3.write(f"{cost_of_living_difference_CAD:,.2f} {option}")
    else:
        col4_3.write(f"ERROR")

    # row 5
    col5_1, col5_2, col5_3 = st.columns(3)
    r_HK_Growth = col5_1.number_input("Cost of living growth in HK percentage", min_value=0.0, max_value=100.0, value=1.0) / 100
    r_CA_Growth = col5_2.number_input("Cost of living growth in Canada percentage", min_value=0.0, max_value=100.0, value=1.0) / 100

    # row 6
    col6_1, col6_2, col6_3 = st.columns(3)
    P_HK_invest = col6_1.number_input("Proportion of of asset on investment percentage in HK", min_value=0.0, max_value=100.0, value=0.0) / 100
    P_CA_invest = col6_2.number_input("Proportion of of asset on investment percentage in Canada", min_value=0.0, max_value=100.0, value=0.0) / 100

    # row 7
    col7_1, col7_2, col7_3 = st.columns(3)
    R_HK_invest = col7_1.number_input("Return of investment (ROI) in percentage in HK", min_value=0.0, max_value=100.0, value=0.0) / 100
    R_CA_invest = col7_2.number_input("Return of investment (ROI) in percentage in Canada", min_value=0.0, max_value=100.0, value=0.0) / 100

case_1 = cpo.cost_of_living_in_Canada(
    S_HK=Salary_HK,
    S_CA_before_tax=Salary_CA,
    R_HK=R_HK_Growth,
    R_CA=R_CA_Growth,
    C_HK=Cost_living_HK,
    C_CA=Cost_living_CA,
    r_HK=r_HK_Growth,
    r_CA=r_CA_Growth,
    rate=exchange_rate,
    percentage_of_investment_HK=P_HK_invest,
    percentage_of_investment_CA=P_CA_invest,
    return_of_investment_HK=R_HK_invest,
    return_of_investment_CA=R_CA_invest
)
st.write("# Opportunity Cost of living in Canada in above setting")
year_wait = st.number_input("number of year expected of living in Canada", min_value=0, max_value=100, value=6)
st.write(f"Opportunity cost: {int(case_1.opportunity_cost_of_living_in_Canada_with_investment(number_of_year=year_wait)):,} HKD")
CA_income_array = np.linspace(40000, 150_000, int((150_000 - 40_000) / 100))
op_with_investment_expected_array = np.array(
    [case_1.opportunity_cost_of_living_in_Canada_with_investment_expected(S_CA_before_tax=i, number_of_year=year_wait) for i in CA_income_array])
# plot graph
chart_data = pd.DataFrame(
    {
        'Canada income before tax': CA_income_array,
        'Opportunity cost': op_with_investment_expected_array,
    }
)
st.line_chart(chart_data, x='Canada income before tax', y='Opportunity cost', x_label='Canada income before tax', y_label='Opportunity cost')

slope, intercept, r_value, p_value, std_err = stats.linregress(CA_income_array, op_with_investment_expected_array)

# break even, opportunity cost = 0
S_CA_break_even = - intercept / slope
st.write(f"When opportunity Cost break even, opportunity cost = 0, salary in Canada before tax: {int(S_CA_break_even):,} CAD")

#
col1, col2 = st.columns(2)
col1.write(f"opportunity Cost when below Canada before tax salary is met")
for i in range(40_000, 100_000, 5000):
    col1.write(
        f"Salary={i} CAD: opportunity cost: "
        f"{int(case_1.opportunity_cost_of_living_in_Canada_with_investment_expected(S_CA_before_tax=i, number_of_year=year_wait)):,} HKD")

col2.write(f"Canada before tax salary to opportunity cost")
for i in range(-500_000, 3_000_000, 100_000):
    col2.write(f"Opportunity Cost = {i / 1e6:.1f}M HKD, Salary = {int((i - intercept) / slope):,} CAD")
