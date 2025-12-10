"""
MATH 5600 Computing Project
Author: Khoa Minh Ngo, Hung Phan Quoc Viet, Markeya Gu
Project: Regression-prediction
"""

using CSV
using DataFrames
using Dates
using Random
using StatsBase
using Polynomials
using Statistics
using LinearAlgebra
using FundamentalsNumericalComputation
using Plots

# ============================================================
# Preparing data 
# ============================================================
data_path = "4182195.csv";
df_raw = CSV.read(data_path, DataFrame);

# Select only the date and tmax to work with
df = select(df_raw, :DATE, :TMAX);

# Filter out rows that has null values (for the current dataset, the null values for this column is just until 1948)
filter!(row -> !ismissing(row.TMAX), df);

# Limit data to only 2025 for faster computation
filter!(row -> row.DATE >= Date(2025, 1, 1), df)

# Rename cols to t and y for convenience
rename!(df, :TMAX => :y);
df.t = 1:nrow(df)

# Check first 5 values of dataframe
first(df, 5)

# Set random seed 
Random.seed!(1234)

N = nrow(df)
idx_all = collect(1:N)

# Doing this to make sure the training set covers the full range for testing to not return NaN for spline method
Ntrain = round(Int, 0.8 * N)
middle_idx = sample(2:N-1, Ntrain - 2; replace = false)
train_idx = vcat(1, middle_idx, N)
test_idx  = setdiff(idx_all, train_idx)

train = df[train_idx, :]
test  = df[test_idx, :]

# Store x, y train/test
x_train = Float64.(train.t)
y_train = Float64.(train.y);

x_test = Float64.(test.t)
y_test = Float64.(test.y);

# Helper for mse and mae metrics
mse(y_pred, y) = mean((y_pred .- y).^2)
mae(y_pred, y) = mean(abs.(y_pred .- y))

n = length(x_train)
println("Total training data points: ", n)
println("Total testing data points: ", length(x_test))
println("")

# ============================================================
# Interpolation methods
# ============================================================

# -------- Polynomial interpolation using barycentric formula --------

# We are choosing a small subset to avoid the polynomial to blow up
k_poly = 4 
subset_idx_poly = round.(Int, range(1, n; length=k_poly))
xt_poly = x_train[subset_idx_poly]
yt_poly = y_train[subset_idx_poly]

ord_poly = sortperm(xt_poly)
xt_poly = xt_poly[ord_poly]
yt_poly = yt_poly[ord_poly]

p = FNC.polyinterp(xt_poly, yt_poly)
y_pred_poly = p.(x_test)

mse_poly = mse(y_pred_poly, y_test)
mae_poly = mae(y_pred_poly, y_test)

println("Testing polynomial interpolation")
println("The MSE error is: ", mse_poly)
println("The MAE error is: ", mae_poly)
println("")

# -------- Spline method --------
ord = sortperm(x_train)
xt = x_train[ord]
yt = y_train[ord]
S = FNC.spinterp(xt, yt)

# Predict on test set
y_pred_spline = S.(x_test)

mse_spline = mse(y_pred_spline, y_test)
mae_spline = mae(y_pred_spline, y_test)

println("Testing spline method")
println("The MSE error is: ", mse_spline);
println("The MAE error is: ", mae_spline);
println("")

# -------- Linear regression --------
A_train = [ones(length(x_train))  x_train]
coefs = A_train \ y_train

# Predict on the test set
A_test = [ones(length(x_test))  x_test]
y_pred_linreg = A_test * coefs

mse_linreg = mse(y_pred_linreg, y_test)
mae_linreg = mae(y_pred_linreg, y_test)

println("Testing linear regression")
println("The MSE error is: ", mse_linreg)
println("The MAE error is: ", mae_linreg)
println("")

# -------- Newton interpolation --------
# The functions below are from formulas from wikipedia: https://en.wikipedia.org/wiki/Newton_polynomial
function newton_divided_differences(x, y)
    a = copy(y)
    n = length(x)

    for j in 2:n # order of difference
        for i in n:-1:j
            a[i] = (a[i] - a[i-1]) / (x[i] - x[i-j+1])
        end
    end
    return a
end

# Evaluate newton interpolation
function newton_eval(x_nodes, a, x)
    n = length(a)
    p = a[n]
    for j in (n-1):-1:1
        p = a[j] .+ (x .- x_nodes[j]) .* p
    end
    return p
end

# Since n is too big, we only take a subset of the train set to actually train
# We will take a subset of k points
k = 5
idx_newton = round.(Int, range(1, n; length=k))
x_train_newton = x_train[idx_newton]
y_train_newton = y_train[idx_newton]

# Compute Newton coefs
a = newton_divided_differences(x_train_newton, y_train_newton)
y_pred_newton = newton_eval(x_train_newton, a, x_test)

mse_newton = mse(y_pred_newton, y_test)
mae_newton = mae(y_pred_newton, y_test)

println("Testing Newton interpolation")
println("The MSE error is: ", mse_newton)
println("The MAE error is: ", mae_newton)
println("")

# ============================================================
# Plotting the result
# ============================================================

# Sort test points for nicer lines
ord_test = sortperm(x_test)
x_test_plot = x_test[ord_test]
y_test_plot = y_test[ord_test]

poly_plot = y_pred_poly[ord_test]
spline_plot = y_pred_spline[ord_test]
linear_plot = y_pred_linreg[ord_test]
newton_plot = y_pred_newton[ord_test]

# -------- Polynomial interpolation --------
plt_poly = scatter(
    x_test_plot, y_test_plot;
    label = "Test data",
    markersize = 4,
    alpha = 0.7,
    xlabel = "Day index in 2025",
    ylabel = "TMAX",
    title = "Polynomial interpolation (degree $(k_poly-1)) vs test data",
)
plot!(plt_poly, x_test_plot, poly_plot; label = "Polynomial fit", linewidth = 2)
display(plt_poly)


# -------- Spline interpolation --------
plt_spline = scatter(
    x_test_plot, y_test_plot;
    label = "Test data",
    markersize = 4,
    alpha = 0.7,
    xlabel = "Day index in 2025",
    ylabel = "TMAX",
    title = "Spline interpolation vs test data",
)
plot!(plt_spline, x_test_plot, spline_plot; label = "Spline fit", linewidth = 2)
display(plt_spline)

# -------- Linear Regression --------
plt_linear = scatter(
    x_test_plot, y_test_plot;
    label = "Test data",
    markersize = 4,
    alpha = 0.7,
    xlabel = "Day index in 2025",
    ylabel = "TMAX",
    title = "Linear regression vs test data",
)
plot!(plt_linear, x_test_plot, linear_plot; label = "Linear fit", linewidth = 2)
display(plt_linear)


# -------- Newton Interpolation --------
plt_newton = scatter(
    x_test_plot, y_test_plot;
    label = "Test data",
    markersize = 4,
    alpha = 0.7,
    xlabel = "Day index in 2025",
    ylabel = "TMAX",
    title = "Newton interpolation (degree $(k-1)) vs test data",
)
plot!(plt_newton, x_test_plot, newton_plot; label = "Newton fit", linewidth = 2)
display(plt_newton)

# -------- Plotting MAE and MSE results --------
models = ["Polynomial", "Spline", "Linear", "Newton"]

mae_vals = [mae_poly, mae_spline, mae_linreg, mae_newton]
mse_vals = [mse_poly, mse_spline, mse_linreg, mse_newton]

plt_mae = bar(
    models, mae_vals;
    xlabel = "Model",
    ylabel = "MAE",
    title = "Test MAE by model",
)
display(plt_mae)

plt_mse = bar(
    models, mse_vals;
    xlabel = "Model",
    ylabel = "MSE",
    title = "Test MSE by model",
)
display(plt_mse)

# Create output directory
mkpath("figures")

# Save all figures
savefig(plt_poly, "figures/plot_polynomial.png")
savefig(plt_spline, "figures/plot_spline.png")
savefig(plt_linear, "figures/plot_linear_regression.png")
savefig(plt_newton, "figures/plot_newton.png")
savefig(plt_mae, "figures/plot_mae.png")
savefig(plt_mse, "figures/plot_mse.png")