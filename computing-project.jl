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

data_path = "4182195.csv";
df_raw = CSV.read(data_path, DataFrame);

# Select only the date and tmax to work with
df = select(df_raw, :DATE, :TMAX);

# Filter out rows that has null values (for the current dataset, the null values for this column is just until 1948)
filter!(row -> !ismissing(row.TMAX), df);
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

mse(y_pred, y) = mean((y_pred .- y).^2)
mae(y_pred, y) = mean(abs.(y_pred .- y))

n = length(x_train)
println("Total training data points: ", n)
println("Total testing data points: ", length(x_test))
println("")

# -------- Polynomial interpolation --------

# Since n is too big, we only take a subset of the train set to actually train
# Take a subset of k points
k = 30
subset_idx = sample(1:n, k; replace = false)
x_train_sub = x_train[subset_idx]
y_train_sub = y_train[subset_idx]

V = [ x_train_sub[i]^j for i=1:k, j=0:k-1 ]

c = V \ y_train_sub
p = Polynomial(c)
y_pred = p.(x_test)

println("Testing polynomial interpolation")
println("The MSE error is: ", mse(y_pred, y_test))
println("The MAE error is: ", mae(y_pred, y_test))
println("")

# -------- Spline method --------
ord = sortperm(x_train)
xt = x_train[ord]
yt = y_train[ord]
S = FNC.spinterp(xt, yt)

# Predict on test set
y_pred = S.(x_test)

println("Testing spline method")
println("The MSE error is: ", mse(y_pred, y_test));
println("The MAE error is: ", mae(y_pred, y_test));
println("")

# -------- Linear regression --------
A_train = [ones(length(x_train))  x_train]
coefs = A_train \ y_train

# Predict on the test set
A_test = [ones(length(x_test))  x_test]
y_pred = A_test * coefs

println("Testing linear regression")
println("The MSE error is: ", mse(y_pred, y_test))
println("The MAE error is: ", mae(y_pred, y_test))
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

# Sort the subset data points to ensure date ordering 
ord_sub = sortperm(x_train_sub)
x_train_newton = x_train_sub[ord_sub]
y_train_newton = y_train_sub[ord_sub]

# Compute Newton coefs
a = newton_divided_differences(x_train_newton, y_train_newton)
y_pred = newton_eval(x_train_newton, a, x_test)

println("Testing Newton interpolation")
println("The MSE error is: ", mse(y_pred, y_test))
println("The MAE error is: ", mae(y_pred, y_test))
println("")
