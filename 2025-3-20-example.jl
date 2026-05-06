# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Julia 1.9.4
#     language: julia
#     name: julia-1.9
# ---

using Revise
using LazySets
using DifferentialEquations
using LazySets
using ProgressMeter
using ProgressBars
using JLD2
using Flux
using LinearAlgebra
using ReverseDiff
using Plots
using Statistics
using Optimisers, ParameterSchedulers
using ModelVerification
using ONNXNaiveNASflux, NaiveNASflux, .NaiveNASlib


# +
# using CUDA

# # Check if GPU is available
# if CUDA.functional()
#     device!(0)
#     CUDA.allowscalar(false)  # Disallow scalar operations on the GPU (optional)
# else
#     println("GPU is not available. Using CPU.")
# end
# -

include("affine_dynamics.jl")
include("dataset.jl")
X = Hyperrectangle(low = [0, 0], high = [4,4])
U = Hyperrectangle(low = [-1], high = [1])
X_unsafe = Hyperrectangle(low = [1.5, 0,0], high = [2.5,2, π])


# +
using TaylorModels
import RobotDynamics

myTaylorModelN(nv::Integer, ord::Integer, x0::IntervalBox{N,T}, dom::IntervalBox{N,T},vars::Vector) where {N,T} =
    TaylorModelN(x0[nv] + vars[nv], zero(dom[1]), x0, dom)
function taylor_model(center, radius, model, u)
    _dim = length(center)

    point = IntervalBox([interval(center[i]) for i in 1:_dim])
    region = IntervalBox([(center[i].-radius[i])..(center[i].+radius[i]) for i in 1:_dim])
    var = set_variables("x", numvars=_dim, order=2)
    taylor_var = [myTaylorModelN(i,1, point,region,var) for i in 1:_dim]
    taylor_var = [TaylorModels.TaylorModelN(i,1, point,region) for i in 1:_dim]
    dyn_x = RobotDynamics.dynamics(model, taylor_var, u)
    
    lower_w = zeros(_dim, _dim)
    upper_w = zeros(_dim, _dim)
    lower_b = zeros(_dim,1)
    upper_b = zeros(_dim,1)
    for i in 1:_dim
        
        if isa(dyn_x[i], TaylorModelN)
            for j in 1:_dim
                lower_w[i, j] = inf(polynomial(dyn_x[i])[1][j])
                upper_w[i, j] = sup(polynomial(dyn_x[i])[1][j])
            end
            lower_b[i, 1] = inf(polynomial(dyn_x[i])[0][1]) + inf(remainder(dyn_x[i])) - sum([lower_w[i, j] .* center[j] for j in 1:_dim])
            upper_b[i, 1] = sup(polynomial(dyn_x[i])[0][1]) + sup(remainder(dyn_x[i])) - sum([upper_w[i, j] .* center[j] for j in 1:_dim])
            
        else
            
            lower_b[i, 1] = dyn_x[i]
            upper_b[i, 1] = dyn_x[i]
        end      
    end
    return lower_w, upper_w, lower_b, upper_b
end

function find_bounds(w, b, lower_x, upper_x)
    lower_x = reshape(lower_x, size(b))
    upper_x = reshape(upper_x, size(b))
    low = clamp.(w, 0, Inf) * lower_x + clamp.(w, -Inf, 0) * upper_x + b
    up = clamp.(w, 0, Inf) * upper_x + clamp.(w, -Inf, 0) * lower_x + b
    return low, up
end
# -

# For ours
include("affine_dynamics.jl")
include("dataset.jl")
include("visualize.jl")


function Base.tanh(A::Matrix{Flux.NilNumber.Nil})
    tanh.(A)
end
using Flux



# 定义网络结构
layer1 = Dense(2, 2, tanh)
layer2 = Dense(2, 2, tanh)
layer3 = Dense(2, 1)

# 手动初始化第一层的权重和偏置
layer1.weight .= [0.2 0.1; 0.2 -0.1]
layer1.bias   .= [0, 0]

layer2.weight .= [0.1 0.2; 0.1 -0.2]
layer2.bias   .= [0, 0]

# 手动初始化第二层的权重和偏置
layer3.weight .= [0.1 0.1]
layer3.bias   .= 0.0

# 组合成模型
original_model = Chain(layer1, layer2, layer3)


phi_model = original_model
# find all the potential root region list, as hyperrectangles
dx = 1
dy = 1
d_theta = 1
α = 0.2
sub_X_list = split(X, [dx, dy])

root_region_list = []
for sub_X in sub_X_list
    v_list = vertices_list(sub_X)
    v_mat = cat(v_list..., dims=length(size(v_list[1])) + 1)
    phi_v_sub = phi_model(v_mat)
    (all(x -> x < 0, phi_v_sub) || all(x -> x > 0, phi_v_sub)) && continue
    push!(root_region_list, sub_X)
end
@show length(root_region_list)


model = RobotZoo.Pendulum()

u_list = vertices_list(U)
u_mat = cat(u_list..., dims=length(size(u_list[1])) + 1)
violated_unknown_region_list = []

#My light-corwn
@showprogress for root_region in root_region_list
    verified_flag = false
    x = root_region.center
    already_hold_sub_region_list = []
    union_all = nothing
    for i in 1:size(u_mat)[2]
        u = u_mat[:, i]
        lower_w, upper_w, lower_b, upper_b = taylor_model(x, root_region.radius, model, u)
        gradient_constraint = HPolyhedron(ones(1, 2), zeros(1))
        search_method = BFS(max_iter=1, batch_size=1)
        split_method = Bisect(1) # must not use inherit_pre_bound
        pre_bound_method = Crown(false, true, true, ModelVerification.zero_slope)
        solver = VeriGrad(false, false, false, pre_bound_method, true, true, Flux.ADAM(0.1), 10, false,[lower_w, upper_w, lower_b, upper_b, RobotDynamics.dynamics(model, x, u),α])
        problem = Problem(phi_model, root_region, gradient_constraint)
        res = verify(search_method, split_method, solver, problem,verbose=false,collect_bound=true)
        pre_bound_method = LightCrown(false, true, true, ModelVerification.zero_slope)
        solver = VeriGrad(false, false, false, pre_bound_method, true, true, Flux.ADAM(0.1), 10, false,[lower_w, upper_w, lower_b, upper_b, RobotDynamics.dynamics(model, x, u),α])
        problem = Problem(phi_model, root_region, gradient_constraint)
        res = verify(search_method, split_method, solver, problem,verbose=false,collect_bound=true)
        for j in eachindex(res.info["verified_bounds"])
            push!(already_hold_sub_region_list, Hyperrectangle(low = res.info["verified_bounds"][j].batch_data_min[1:end-1, 1], high = res.info["verified_bounds"][j].batch_data_max[1:end-1,1]))
        end
        if length(already_hold_sub_region_list) > 1
            union_all = UnionSetArray([already_hold_sub_region_list[i] for i in eachindex(already_hold_sub_region_list)])
        else
            if length(already_hold_sub_region_list) > 0
                union_all = already_hold_sub_region_list[1]
            end
        end
        if (res.status == :holds) || ((length(already_hold_sub_region_list) > 0 && (root_region ⊆ union_all)))
            verified_flag = true
            break
        end
    end
    verified_flag || push!(violated_unknown_region_list, root_region)
end
@show length(violated_unknown_region_list)
@show suc_rate = (length(root_region_list) - length(violated_unknown_region_list)) / length(root_region_list)

#Ours 
@showprogress for root_region in root_region_list
    verified_flag = false
    x = root_region.center
    already_hold_sub_region_list = []
    union_all = nothing
    for i in 1:size(u_mat)[2]
        u = u_mat[:, i]
        lower_w, upper_w, lower_b, upper_b = taylor_model(x, root_region.radius, model, u)
        gradient_constraint = HPolyhedron(ones(1, 2), zeros(1))
        search_method = BFS(max_iter=1, batch_size=1)
        split_method = Bisect(1) # must not use inherit_pre_bound
        pre_bound_method = Crown(false, true, true, ModelVerification.zero_slope)
        solver = VeriGrad(false, false, false, pre_bound_method, true, true, Flux.ADAM(0.1), 10, false,[lower_w, upper_w, lower_b, upper_b, RobotDynamics.dynamics(model, x, u),α])
        problem = Problem(phi_model, root_region, gradient_constraint)
        res = verify(search_method, split_method, solver, problem,verbose=false,collect_bound=true)
        for j in eachindex(res.info["verified_bounds"])
            push!(already_hold_sub_region_list, Hyperrectangle(low = res.info["verified_bounds"][j].batch_data_min[1:end-1, 1], high = res.info["verified_bounds"][j].batch_data_max[1:end-1,1]))
        end
        if length(already_hold_sub_region_list) > 1
            union_all = UnionSetArray([already_hold_sub_region_list[i] for i in eachindex(already_hold_sub_region_list)])
        else
            if length(already_hold_sub_region_list) > 0
                union_all = already_hold_sub_region_list[1]
            end
        end
        if (res.status == :holds) || ((length(already_hold_sub_region_list) > 0 && (root_region ⊆ union_all)))
            verified_flag = true
            break
        end
    end
    verified_flag || push!(violated_unknown_region_list, root_region)
end
@show length(violated_unknown_region_list)
@show suc_rate = (length(root_region_list) - length(violated_unknown_region_list)) / length(root_region_list)
# -

# # My CBF Verify

# +
# For My CBF Verify menthod
include("affine_dynamics.jl")
include("dataset.jl")
include("visualize.jl")


model_state = JLD2.load("car_naive_model_1_0_0.1_pgd_relu_20.jld2", "model_state");
# model_state = JLD2.load("car_naive_small_model_1_0_1_pgd_relu_20.jld2", "model_state");
# model_state = JLD2.load("car_naive_big_model_1_0_1_pgd_relu_20.jld2", "model_state");
# model_state = JLD2.load("car_adv20_model_1_0_0.1_pgd_relu_20.jld2", "model_state");
# model_state = JLD2.load("car_adv20_small_model_1_0_1_pgd_relu_20.jld2", "model_state");
# model_state = JLD2.load("car_adv20_big_model_1_0_1_pgd_relu_20.jld2", "model_state");


original_model = Chain(
    Dense(3 => 16, relu),   # activation function inside layer
    Dense(16 => 64, relu),   # activation function inside layer
    Dense(64 => 16, relu),   # activation function inside layer
    Dense(16 => 1)
)

# original_model = Chain(
#     Dense(3 => 8, relu),   # activation function inside layer
#     Dense(8 => 8, relu),   # activation function inside layer
#     Dense(8 => 8, relu),   # activation function inside layer
#     Dense(8 => 1)
# )

# original_model = Chain(
#     Dense(3 => 64, relu),   # activation function inside layer
#     Dense(64 => 128, relu),   # activation function inside layer
#     Dense(128 => 64, relu),   # activation function inside layer
#     Dense(64 => 1)
# )

Flux.loadmodel!(original_model, model_state);
phi_model = original_model
# find all the potential root region list, as hyperrectangles
dx = 100
dy = 100
d_theta = 100
α = 0.5
sub_X_list = split(X, [dx, dy, d_theta])

root_region_list = []
for sub_X in sub_X_list
    v_list = vertices_list(sub_X)
    v_mat = cat(v_list..., dims=length(size(v_list[1])) + 1)
    phi_v_sub = phi_model(v_mat)
    (all(x -> x < 0, phi_v_sub) || all(x -> x > 0, phi_v_sub)) && continue
    push!(root_region_list, sub_X)
end
@show length(root_region_list)


model = RobotZoo.DubinsCar()

u_list = vertices_list(U)
u_mat = cat(u_list..., dims=length(size(u_list[1])) + 1)
violated_unknown_region_list = []
