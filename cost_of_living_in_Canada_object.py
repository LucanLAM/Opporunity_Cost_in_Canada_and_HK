import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import streamlit as st
import pandas as pd


def cumulative_sum(S, R, N):
    return (S / R) * ((1 + R) ** N - 1)


def income_before_to_income_after_tax(income_before_tax):
    print("Warning! The income after tax is a rough estimation")
    return 0.6178712665406427 * income_before_tax + 10057.770510396978


def income_after_tax_to_income_before_tax(income_after_tax):
    print("Warning! The income before tax is a rough estimation")
    return (income_after_tax - 10057.770510396978) / 0.6178712665406427


def plot_data_matplotlib(data_x, data_y, xlabel, ylabel, title):
    # Creating the plot
    plt.plot(data_x, data_y, marker='o')

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display the plot
    plt.show()


def plot_data_streamlit(data_x, data_y, xlabel, ylabel, title):
    # Creating the plot
    chart_data = pd.DataFrame(
        {
            xlabel: data_x,
            ylabel: data_y,
        }
    )

    # Adding labels and title
    st.title(title)
    st.line_chart(chart_data, x=xlabel, y=ylabel, x_label=xlabel, y_label=ylabel)


def stat_analyze(x_array, y_array, overhead=None):
    if not (overhead is None):
        print(overhead)
    # Perform linear regression using scipy's linregress function
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
    # Print the results of the regression
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-squared: {r_value ** 2}")
    print(f"P-value: {p_value}")
    print(f"Standard Error: {std_err}")


# def discrete_convolve(f, g, n_values):
#     result = np.zeros_like(n_values)
def support_convolve(Rr, Pi, Ri, N):
    temp_sum = 0
    for i in range(1, N + 1):
        # print(f"i = {i}")
        temp_sum += (1 + Rr) ** (i - 1) * (1 + Pi * Ri) ** (N - i)
    return temp_sum


class cost_of_living_in_Canada:
    def __init__(self, S_HK, S_CA_before_tax, R_HK, R_CA, C_HK, C_CA, r_HK, r_CA, rate, Canada_income_income_after_tax_pair=None, percentage_of_investment_HK=0.0,
                 return_of_investment_HK=0.0, percentage_of_investment_CA=0.0, return_of_investment_CA=0.0):
        if Canada_income_income_after_tax_pair is None:
            self.Canada_income_income_after_tax_pair = {
                40000: 32978,
                45000: 36424,
                50000: 39719,
                55000: 43265,
                60000: 46559,
                62500: 48168,
                65000: 49801,
                67500: 51443,
                70000: 53155,
                72500: 54914,
                75000: 56523,
                77500: 58282,
                80000: 60040,
                85000: 63558,
                90000: 67075,
                95000: 70567,
                100000: 73993,
                110000: 80662,
                150000: 103429,
                200000: 130088
            }
        else:
            self.Canada_income_income_after_tax_pair = Canada_income_income_after_tax_pair
        self.S_HK = S_HK
        self.S_CA = self.Canada_income_after_tax(S_CA_before_tax)
        self.R_HK = R_HK
        self.R_CA = R_CA
        self.C_HK = C_HK
        self.C_CA = C_CA
        self.r_HK = r_HK
        self.r_CA = r_CA
        self.rate = rate
        self.percentage_of_investment_HK = percentage_of_investment_HK
        self.return_of_investment_HK = return_of_investment_HK
        self.percentage_of_investment_CA = percentage_of_investment_CA
        self.return_of_investment_CA = return_of_investment_CA

    def cost_of_living_in_Canada(self, number_of_year):
        net_HK = cumulative_sum(self.S_HK, self.R_HK, number_of_year) - cumulative_sum(self.C_HK, self.r_HK, number_of_year)
        net_CA = cumulative_sum(self.S_CA, self.R_CA, number_of_year) - cumulative_sum(self.C_CA, self.r_CA, number_of_year)
        return net_HK - net_CA

    def Canada_income_after_tax(self, income_before_tax):
        x = list(self.Canada_income_income_after_tax_pair.keys())
        # y = list(self.Canada_income_income_after_tax_pair.values())
        if income_before_tax in x:
            return self.Canada_income_income_after_tax_pair[income_before_tax]
        else:
            lower_key = None
            upper_key = None
            for i in sorted(x):
                if i < income_before_tax:
                    lower_key = i
                elif i > income_before_tax:
                    upper_key = i
                    break

            if lower_key is None or upper_key is None:
                return income_before_to_income_after_tax(income_before_tax)
                # raise ValueError(f"income before tax {income_before_tax} is out of the bounds of the data range.")

            lower_value = self.Canada_income_income_after_tax_pair[lower_key]
            upper_value = self.Canada_income_income_after_tax_pair[upper_key]

            # interpolation
            interpolated_value = lower_value + (income_before_tax - lower_key) * (upper_value - lower_value) / (upper_key - lower_key)
            return int(interpolated_value)

    def Canada_income_after_tax_inverse(self, income_after_tax):
        x = list(self.Canada_income_income_after_tax_pair.keys())
        x = sorted(x)
        # y = list(Canada_income_income_after_tax_pair.values())
        if income_after_tax < self.Canada_income_income_after_tax_pair[x[0]] or \
                income_after_tax > self.Canada_income_income_after_tax_pair[x[-1]]:
            return income_after_tax_to_income_before_tax(income_after_tax)
            # raise ValueError(
            #     f"Income after tax {income_after_tax} is out of data range, "
            #     f"min = {self.Canada_income_income_after_tax_pair[x[0]]}, max = {self.Canada_income_income_after_tax_pair[x[-1]]}")

        for key in x:
            if self.Canada_income_income_after_tax_pair[key] == income_after_tax:
                return key

        lower_key = None
        upper_key = None
        for i in range(1, len(x)):
            if self.Canada_income_income_after_tax_pair[x[i - 1]] <= income_after_tax \
                    <= self.Canada_income_income_after_tax_pair[x[i]]:
                lower_key = x[i - 1]
                upper_key = x[i]
                break
        lower_value = self.Canada_income_income_after_tax_pair[lower_key]
        upper_value = self.Canada_income_income_after_tax_pair[upper_key]
        # interpolation
        interpolated_value = lower_key + (income_after_tax - lower_value) * (upper_key - lower_key) / (upper_value - lower_value)
        return int(interpolated_value)

    def slope_OP(self, number_of_year):
        return - (self.rate * ((1 + self.R_CA) ** number_of_year - 1)) / self.R_CA

    def intercept_OP(self, number_of_year):
        return cumulative_sum(self.S_HK, self.R_HK, number_of_year) - cumulative_sum(self.C_HK, self.r_HK, number_of_year) + self.rate * cumulative_sum(self.C_CA, self.r_CA,
                                                                                                                                                        number_of_year)

    def income_accept_cost(self, number_of_year, acceptable_cost):
        after_tax = (acceptable_cost - self.intercept_OP(number_of_year=number_of_year)) / self.slope_OP(number_of_year=number_of_year)
        return self.Canada_income_after_tax_inverse(income_after_tax=after_tax)

    def opportunity_cost(self, expected_S_CA, number_of_year):
        after_tax = self.Canada_income_after_tax(expected_S_CA)
        return after_tax * self.slope_OP(number_of_year=number_of_year) + self.intercept_OP(number_of_year=number_of_year)

    def annual_wage_HK(self, year):
        return self.S_HK * (1 + self.R_HK) ** (year - 1)

    def annual_cost_of_living_HK(self, year):
        return self.C_HK * (1 + self.r_HK) ** (year - 1)

    def annual_wage_CA(self, year):
        return self.S_CA * (1 + self.R_CA) ** (year - 1)

    def annual_cost_of_living_CA(self, year):
        return self.C_CA * (1 + self.r_CA) ** (year - 1)

    def annual_net_HK(self, year):
        return self.annual_wage_HK(year=year) - self.annual_cost_of_living_HK(year=year)

    def annual_net_CA(self, year):
        return self.annual_wage_CA(year=year) - self.annual_cost_of_living_CA(year=year)

    def cumulative_sum_with_investment_HK(self, number_of_year):
        cumulative_sum_invest = self.annual_net_HK(year=1)
        # print(f"cumulative_sum_invest first = {cumulative_sum_invest}")
        # if number_of_year > 1:
        for i in range(2, number_of_year + 1):
            cumulative_sum_invest = cumulative_sum_invest + cumulative_sum_invest * self.percentage_of_investment_HK * self.return_of_investment_HK + self.annual_net_HK(year=i)
            # print(f"i={i}, cumulative_sum_invest={cumulative_sum_invest}")
        return cumulative_sum_invest

    def cumulative_sum_with_investment_CA(self, number_of_year):
        cumulative_sum_invest = self.annual_net_CA(year=1)
        # print(f"cumulative_sum_invest first = {cumulative_sum_invest}")
        # if number_of_year > 1:
        for i in range(2, number_of_year + 1):
            cumulative_sum_invest = cumulative_sum_invest + cumulative_sum_invest * self.percentage_of_investment_CA * self.return_of_investment_CA + self.annual_net_CA(year=i)
            # print(f"i={i}, cumulative_sum_invest={cumulative_sum_invest}")
        return cumulative_sum_invest

    def opportunity_cost_of_living_in_Canada_with_investment(self, number_of_year):
        return self.cumulative_sum_with_investment_HK(number_of_year=number_of_year) - self.rate * self.cumulative_sum_with_investment_CA(number_of_year=number_of_year)

    def annual_wage_CA_expected(self, S_CA_before_tax, year):
        S_CA_after_tax = self.Canada_income_after_tax(S_CA_before_tax)
        return S_CA_after_tax * (1 + self.R_CA) ** (year - 1)

    def annual_net_CA_expected(self, S_CA_before_tax, year):
        return self.annual_wage_CA_expected(S_CA_before_tax=S_CA_before_tax, year=year) - self.annual_cost_of_living_CA(year=year)

    def cumulative_sum_with_investment_CA_expected(self, S_CA_before_tax, number_of_year):
        cumulative_sum_invest = self.annual_net_CA_expected(S_CA_before_tax=S_CA_before_tax, year=1)
        # print(f"cumulative_sum_invest first = {cumulative_sum_invest}")
        # if number_of_year > 1:
        for i in range(2, number_of_year + 1):
            cumulative_sum_invest = cumulative_sum_invest + cumulative_sum_invest * self.percentage_of_investment_CA * self.return_of_investment_CA + \
                                    self.annual_net_CA_expected(S_CA_before_tax=S_CA_before_tax, year=i)
            # print(f"i={i}, cumulative_sum_invest={cumulative_sum_invest}")
        return cumulative_sum_invest

    def opportunity_cost_of_living_in_Canada_with_investment_expected(self, S_CA_before_tax, number_of_year):
        return self.cumulative_sum_with_investment_HK(number_of_year=number_of_year) - self.rate * self.cumulative_sum_with_investment_CA_expected(S_CA_before_tax=S_CA_before_tax,
                                                                                                                                                   number_of_year=number_of_year)

    def slope_invest(self, number_of_year):
        return - support_convolve(Rr=self.R_CA, Pi=self.percentage_of_investment_CA, Ri=self.return_of_investment_CA, N=number_of_year)

    def intercept_invest(self, number_of_year):
        temp = self.S_HK * support_convolve(Rr=self.R_HK, Pi=self.percentage_of_investment_HK, Ri=self.return_of_investment_HK, N=number_of_year)
        temp -= self.C_HK * support_convolve(Rr=self.r_HK, Pi=self.percentage_of_investment_HK, Ri=self.return_of_investment_HK, N=number_of_year)
        temp += self.C_CA * support_convolve(Rr=self.r_CA, Pi=self.percentage_of_investment_CA, Ri=self.return_of_investment_CA, N=number_of_year)
        return temp


if __name__ == '__main__':
    case1 = cost_of_living_in_Canada(
        S_HK=25000 * 12,
        S_CA_before_tax=55000,
        R_HK=0.03,
        R_CA=0.02,
        C_HK=10000 * 12,
        C_CA=3000 * 12,
        r_HK=0.01,
        r_CA=0.01,
        rate=5.7,
        percentage_of_investment_HK=0.8,
        percentage_of_investment_CA=0.8,
        return_of_investment_HK=0.1,
        return_of_investment_CA=0.1
    )
    number_of_year_of_invest = 6
    a = case1.cumulative_sum_with_investment_HK(number_of_year=number_of_year_of_invest)
    print(f"invest in HK in {number_of_year_of_invest} years = {int(a)} HKD")
    b = case1.cumulative_sum_with_investment_CA(number_of_year=number_of_year_of_invest)
    print(f"invest in CA in {number_of_year_of_invest} years = {int(b)} CAD")
    number_of_year_of_getting_living_in_Canada = 6
    # acceptable_cost = 500_000
    acceptable_cost = 0
    income_cad = case1.income_accept_cost(number_of_year=number_of_year_of_getting_living_in_Canada, acceptable_cost=acceptable_cost)
    print(f"With acceptable cost of {acceptable_cost} HKD in the period of {number_of_year_of_getting_living_in_Canada} years, you need to have annual income of {int(income_cad)} CAD")
    S_CA_expected_before_tax = 30000
    op_cost = case1.opportunity_cost(expected_S_CA=S_CA_expected_before_tax, number_of_year=number_of_year_of_getting_living_in_Canada)
    print(f"opportunity cost with {S_CA_expected_before_tax} CAD wage is: {int(op_cost)}")
    # print(income_cad)
    year_array = np.linspace(1, 10 + 1, dtype=int)
    op_with_investment = np.array([case1.opportunity_cost_of_living_in_Canada_with_investment(number_of_year=i) for i in year_array])
    # plot_data(data_x=year_array, data_y=op_with_investment, xlabel='year', ylabel='Opportunity cose', title='Opportunity cose against year')
    CA_income_array = np.linspace(40000, 150_000, 500)
    op_with_investment_expected_array = np.array([case1.opportunity_cost_of_living_in_Canada_with_investment_expected(S_CA_before_tax=i, number_of_year=6) for i in CA_income_array])
    plot_data_matplotlib(data_x=CA_income_array, data_y=op_with_investment_expected_array, xlabel='Canada income before tax', ylabel='Opportunity cost',
                         title='Opportunity cost against '
                               'income')

    stat_analyze(CA_income_array, op_with_investment_expected_array, overhead='data analyze on CA_income and op_with_investment')
    print(f"slope = {case1.slope_invest(number_of_year=6)}")
    print(f"intercept = {case1.intercept_invest(number_of_year=6)}")
