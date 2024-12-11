using Random
using LinearAlgebra
using Plots
using Dates  # For more detailed timing info
using Distributions
using BenchmarkTools
using Base.Threads
using Serialization
using LambertW
# Upper Confidence Bound Algorithm Implementation
struct UpperConfidenceBound
    n_arms::Int32
    counts::Vector{Int32}
    values::Vector{Float32}
    total_pulls::Vector{Int32}
    reward::Vector{Float32}
end

function UpperConfidenceBound(n_arms)
    UpperConfidenceBound(n_arms, zeros(Int32,n_arms), zeros(Float32,n_arms), zeros(Int32,1), zeros(Float32,1))
end

function select_arm(ucb::UpperConfidenceBound)
    ucb_values = Vector{Float32}(undef, ucb.n_arms)
    total_pulls_log :: Float32 = 2*Float32(2.1) * log(ucb.total_pulls[1])  # Precompute log to avoid multiple calculations
    average_reward :: Float32 = 0.0
    confidence_interval :: Float32 = 0.0
    @inbounds for arm in 1:ucb.n_arms
        if ucb.counts[arm] == 0
            return arm  # If an arm hasn't been pulled, select it
        end
        ucb_values[arm] = ucb.values[arm] - sqrt((total_pulls_log) / ucb.counts[arm])
    end
    return argmin(ucb_values)
end

function update!(ucb::UpperConfidenceBound, chosen_arm::Int32, reward::Float32)
    ucb.counts[chosen_arm] = ucb.counts[chosen_arm] +1
    ucb.total_pulls[1] = ucb.total_pulls[1] +1
    ucb.reward[1] = ucb.reward[1]+reward
    ucb.values[chosen_arm] = ((ucb.counts[chosen_arm] - Float32(1.0)) / ucb.counts[chosen_arm]) * ucb.values[chosen_arm] + (Float32(1.0) / ucb.counts[chosen_arm]) * reward
end

# EXP3 Algorithm Implementation
struct EXP3
    num_arms::Int32
    weights::Vector{Float32}
    probs::Vector{Float32}
    counts::Vector{Int32}
    reward::Vector{Float32}
    cumu_reward::Vector{Float32}
    importance_estimator::Vector{Float32}
end

function EXP3(num_arms)
    EXP3(num_arms, ones(Float32,num_arms), ones(Float32,num_arms) ./ num_arms, zeros(Int32,num_arms), zeros(Float32,1), zeros(Float32,num_arms),zeros(Float32,num_arms))
end

# EXP3+ Algorithm Implementation
struct EXP3plus
    num_arms::Int32
    weights::Vector{Float32}
    probs::Vector{Float32}
    realprobs::Vector{Float32}
    counts::Vector{Int32}
    reward::Vector{Float32}
    cumu_reward::Vector{Float32}
    unajusted_reward::Vector{Float32}
    hedging_weights::Vector{Float32}
end

function EXP3plus(num_arms)
    EXP3plus(num_arms, ones(Float32,num_arms), ones(Float32,num_arms) ./ num_arms, ones(Float32,num_arms) ./ num_arms, zeros(Int32,num_arms), zeros(Float32,1), zeros(Float32,num_arms),zeros(Float32,num_arms),zeros(Float32,num_arms))
end
function select_arm(exp3::EXP3)
    sum_weights :: Float32 = sum(exp3.weights)  # Avoid recomputing this in the loop
    exp3.probs .= exp3.weights ./ sum_weights
    return rand(Categorical(exp3.probs))  # Categorical sampling
end

function select_arm(exp3plus::EXP3plus)
    sum_weights :: Float32 = sum(exp3plus.weights)  # Avoid recomputing this in the loop
    exp3plus.probs .= exp3plus.weights ./ sum_weights
    sum_hedge_weights :: Float32 = sum(exp3plus.hedging_weights)  # Avoid recomputing this in the loop
    exp3plus.realprobs .= (1.0- sum_hedge_weights) * exp3plus.probs .+ exp3plus.hedging_weights
    return rand(Categorical(exp3plus.realprobs))  # Categorical sampling
end

function update!(exp3::EXP3, chosen_arm::Int32, reward::Float32, eta::Float32, method::String)
    @inbounds begin
        exp3.counts[chosen_arm] += 1
        exp3.reward[1] += reward
        estimated_reward = reward / exp3.probs[chosen_arm]
        exp3.cumu_reward[chosen_arm] += estimated_reward
        if method == "FTRL"
            @simd  for i in 1:exp3.num_arms
                @inbounds exp3.weights[i] = exp(exp3.cumu_reward[i] * -eta)
            end
        else
            exp3.weights[chosen_arm] *= exp(-eta * estimated_reward)
        end
    end
end

function update!(exp3plus::EXP3plus, chosen_arm::Int32, reward::Float32, eta::Float32, method::String,round::Int64)
    @inbounds begin
        exp3plus.counts[chosen_arm] += 1
        exp3plus.reward[1] += reward
        exp3plus.unajusted_reward[chosen_arm] += reward
        gap = zeros(Float32,exp3plus.num_arms)
        ucb = zeros(Float32,exp3plus.num_arms)
        lcb = zeros(Float32,exp3plus.num_arms)
        ucb_inf = Inf 
        if method == "Later"            
            estimated_reward = reward / exp3plus.realprobs[chosen_arm]
            exp3plus.cumu_reward[chosen_arm] += estimated_reward            
            @simd  for i in 1:exp3plus.num_arms
                @inbounds exp3plus.weights[i] = exp(exp3plus.cumu_reward[i] * -eta)
            end
            @simd  for i in 1:exp3plus.num_arms
                @inbounds ucb[i] = min(1.0,  exp3plus.unajusted_reward[i]/exp3plus.counts[i] + sqrt(3*log(round*exp3plus.num_arms^(1/3))/(2*exp3plus.counts[i])))
                @inbounds lcb[i] = max(0.0,  exp3plus.unajusted_reward[i]/exp3plus.counts[i] - sqrt(3*log(round*exp3plus.num_arms^(1/3))/(2*exp3plus.counts[i])))
            end
            @simd  for i in 1:exp3plus.num_arms
               if ucb[i] < ucb_inf
                    ucb_inf = ucb[i]
               end
            end
            @simd  for i in 1:exp3plus.num_arms
                @inbounds gap[i] = max(0.0, -ucb_inf + lcb[i])
                @inbounds exp3plus.hedging_weights[i] = min(1/(2*exp3plus.num_arms),1/2*sqrt(log(exp3plus.num_arms)/(round*exp3plus.num_arms)),256*log(round)/(round*(gap[i])^2))
            end
        end               
    end
end
# Running UCB Algorithm
function run_ucb_algorithm(tid::Int32,monte::Int32,n_arms::Int32, n_rounds::Int32, reward_distributions::Vector{Float32}, rewards::Vector{Float32}, pulls::Matrix{Float32},name::String,randomness::Vector{Float32})
    ucb = UpperConfidenceBound(n_arms)
    chosen_arm ::Int32 = 0
    reward ::Float32 = 0.0
    for i in 1:n_rounds
        if i % 1000000 == 0
            println("Thread $tid in Monte Carlo iteration $monte inner round $i of task $name")
        end  
        chosen_arm = select_arm(ucb)
        reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
        update!(ucb, chosen_arm, reward)
        rewards[i] += ucb.reward[1]
        pulls[i, :] .= pulls[i, :] .+ ucb.counts
    end
end

# Running EXP3 Algorithm
function run_exp3_algorithm(tid::Int32, monte::Int32,n_arms::Int32, n_rounds::Int32, reward_distributions::Vector{Float32}, type::Int32, rewards::Vector{Float32}, pulls::Matrix{Float32},name::String,randomness::Vector{Float32},importance_estimator::Matrix{Float32}, weights::Matrix{Float32}, test_estimator::Matrix{Float32})
    exp3 = EXP3(n_arms)
    eta::Float32 = 2*sqrt(log(n_arms)/n_rounds)
    chosen_arm::Int32 = 0
    reward::Float32 = 0.0
    estimator = zeros(Float32,exp3.num_arms)
    for i in 1:n_rounds
        chosen_arm = select_arm(exp3)
        weights[i,:] = exp3.probs .+ 0.0
        reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
        if i % 1000000 == 0
            println("Thread $tid in Monte Carlo iteration $monte inner round $i of task $name")
        end     
        if type == 1
            update!(exp3, chosen_arm, reward, Float32(2*sqrt(log(n_arms)/i)), "FTRL")
        elseif type == 2
            update!(exp3, chosen_arm, reward, Float32(2*sqrt(log(n_arms)/i)), "OMD")
        else
            update!(exp3, chosen_arm, reward, eta, "OMD")
        end
        rewards[i] += exp3.reward[1]
        pulls[i, :] .= pulls[i, :] .+ exp3.counts
        importance_estimator[i,:] = exp3.cumu_reward .- estimator .+ 0.0
        estimator = exp3.cumu_reward .+ 0.0
        test_estimator[i,:] = exp3.cumu_reward./ i
    end
end

# Running EXP3plus Algorithm
function run_exp3plus_algorithm(tid::Int32,monte::Int32, n_arms::Int32, n_rounds::Int32, reward_distributions::Vector{Float32}, rewards::Vector{Float32}, pulls::Matrix{Float32},name::String,randomness::Vector{Float32},importance_estimator::Matrix{Float32}, weights::Matrix{Float32}, test_estimator::Matrix{Float32})
    exp3plus = EXP3plus(n_arms)    
    chosen_arm::Int32 = 0
    reward::Float32 = 0.0
    estimator = zeros(Float32,exp3plus.num_arms)
    for i in 1:n_rounds      
        if i % 1000000 == 0
            println("Thread $tid in Monte Carlo iteration $monte inner round $i of task $name")
        end     
        if i> exp3plus.num_arms
            chosen_arm = select_arm(exp3plus)
            weights[i,:] = exp3plus.realprobs .+ 0.0
            reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
            update!(exp3plus, chosen_arm, reward, Float32(0.5*sqrt(log(n_arms)/(i*n_arms))),"Later",i)
            rewards[i] += exp3plus.reward[1]
            pulls[i, :] .= pulls[i, :] .+ exp3plus.counts
        else
            chosen_arm = i 
            weights[i,:] = exp3plus.realprobs .+ 0.0
            reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
            update!(exp3plus, chosen_arm, reward, Float32(0.5*sqrt(log(n_arms)/(i*n_arms))),"Initial",i)
            rewards[i] += exp3plus.reward[1]
            pulls[i, :] .= pulls[i, :] .+ exp3plus.counts
        end
        importance_estimator[i,:] = exp3plus.cumu_reward .- estimator .+ 0.0
        estimator = exp3plus.cumu_reward .+ 0.0
        test_estimator[i,:] = exp3plus.cumu_reward./ i
    end
end

# Tsallis entropy algorithm 
struct Tsallis
    num_arms::Int32
    probs::Vector{Float32}
    counts::Vector{Int32}
    reward::Vector{Float32}
    cumu_reward::Vector{Float32}
    importance_estimator::Vector{Float32}
    warmup::Vector{Float32}
end
function Tsallis(num_arms)
    Tsallis(num_arms, ones(Float32,num_arms) ./ num_arms, zeros(Int32,num_arms), zeros(Float32,1), zeros(Float32,num_arms),zeros(Float32,num_arms), ones(Float32,1)*Float32(-0.1))
end
function select_arm(tsallis::Tsallis)
    return rand(Categorical(tsallis.probs))  # Categorical sampling
end

function update!(tsallis::Tsallis, chosen_arm::Int32, reward::Float32, eta::Float32)
    @inbounds begin
        tsallis.counts[chosen_arm] += 1
        tsallis.reward[1] += reward
        estimated_reward = reward / tsallis.probs[chosen_arm]
        tsallis.cumu_reward[chosen_arm] += estimated_reward
        tsallis.probs .= newton!(tsallis, tsallis.warmup[1], eta).+0.0    
        # println("1234",tsallis.probs)      
    end   
end
function newton!(tsallis::Tsallis, initial::Float32, eta::Float32)
    temp = zeros(Float32,tsallis.num_arms)
    prev = ones(Float32,tsallis.num_arms)

    # println(tsallis.cumu_reward)
    # if sum(tsallis.cumu_reward)< 1e-9
    #     return ones(Float32,tsallis.num_arms)./ tsallis.num_arms
    # end
    @inbounds begin
        while abs(sum(temp)-1.0) > 1e-6            
            @simd for i in 1:tsallis.num_arms
                @inbounds temp[i] = 4.0*(eta*(tsallis.cumu_reward[i] - initial))^(-2)
            end
            # println(sum(temp))
            # println(abs(sum(temp)-1.0))
            initial = initial - (sum(temp) - 1.0)/(eta*sum(temp.^(3/2)))   
            # println(initial)         
        end
    end
    for i in 1:tsallis.num_arms
        temp[i] = 4*(eta*(tsallis.cumu_reward[i] - initial))^(-2)
    end 
    tsallis.warmup[1] = initial
    return temp ./ sum(temp)
end

# Running Tsallisinf Algorithm
function run_Tsallisinf_algorithm(tid::Int32, monte::Int32,n_arms::Int32, n_rounds::Int32, reward_distributions::Vector{Float32}, rewards::Vector{Float32}, pulls::Matrix{Float32},name::String,randomness::Vector{Float32},importance_estimator::Matrix{Float32}, weights::Matrix{Float32}, test_estimator::Matrix{Float32})
    tsallis = Tsallis(n_arms)
    chosen_arm::Int32 = 0
    reward::Float32 = 0.0
    estimator = zeros(Float32,tsallis.num_arms)
    for i in 1:n_rounds
        chosen_arm = select_arm(tsallis)
        weights[i,:] = tsallis.probs .+ 0.0
        reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
        if i % 1000000 == 0
            println("Thread $tid in Monte Carlo iteration $monte inner round $i of task $name")
        end     
        update!(tsallis, chosen_arm, reward, Float32(2*sqrt(1.0/i)))
        rewards[i] += tsallis.reward[1]
        pulls[i, :] .= pulls[i, :] .+ tsallis.counts
        importance_estimator[i,:] = tsallis.cumu_reward .- estimator .+ 0.0
        estimator = tsallis.cumu_reward .+ 0.0
        test_estimator[i,:] = tsallis.cumu_reward./ i
    end
end



# Shinji's log-barrier
struct Shinji
    num_arms::Int32
    probs::Vector{Float32}
    counts::Vector{Int32}
    reward::Vector{Float32}
    cumu_reward::Vector{Float32}
    importance_estimator::Vector{Float32}
    warmup::Vector{Float32}
    gamma :: Vector{Float32}
    m :: Vector{Float32}
    B :: Float32
    eta :: Float32
end
function Shinji(num_arms)
    Shinji(num_arms, ones(Float32,num_arms) ./ num_arms, zeros(Int32,num_arms), zeros(Float32,1), zeros(Float32,num_arms),zeros(Float32,num_arms), ones(Float32,1), ones(Float32,num_arms).* Float32(2.0), ones(Float32,num_arms).* Float32(0.5), Float32(1/(log(n_rounds))), Float32(1/4))
end
function select_arm(shinji::Shinji)
    return rand(Categorical(shinji.probs))  # Categorical sampling
end

function update!(shinji::Shinji)
    shinji.probs .= newton!(shinji).+0.0
end
function newton!(shinji::Shinji)
    temp = zeros(Float32,shinji.num_arms)
    temp1 = zeros(Float32,shinji.num_arms)
    initial = minimum(shinji.cumu_reward.+shinji.m)-1e-1
    @inbounds begin
        while abs(sum(temp)-1.0) > 1e-6
            
            @simd for i in 1:shinji.num_arms
                @inbounds temp[i] = shinji.gamma[i]/(shinji.cumu_reward[i]+shinji.m[i] - initial)
                @inbounds temp1[i] = shinji.gamma[i]/(shinji.cumu_reward[i]+shinji.m[i] - initial)^2
            end
            initial = initial - (sum(temp) - 1.0)/(sum(temp1))
        end
    end
    @simd for i in 1:shinji.num_arms
        @inbounds temp[i] = shinji.gamma[i]/(shinji.cumu_reward[i]+shinji.m[i] - initial)
    end
    # print(temp)
    return temp./ sum(temp) 
end
function update!!(shinji::Shinji, chosen_arm::Int32, reward::Float32)
    @inbounds begin
        shinji.counts[chosen_arm] += 1
        shinji.reward[1] += reward
        for i in 1:shinji.num_arms
            shinji.importance_estimator[i] = shinji.m[i] 
            if i == chosen_arm
                shinji.importance_estimator[i] += (reward-shinji.m[i])/shinji.probs[i]
            end
        end
        shinji.cumu_reward .= shinji.cumu_reward .+ shinji.importance_estimator.+0.0
        v = zeros(Float32,shinji.num_arms)
        for i in 1:shinji.num_arms
            if i == chosen_arm
                v[i] = (reward - shinji.m[i])^(2)*(1-shinji.probs[i])^2
            else
                v[i] = (reward - shinji.m[chosen_arm])^(2)*(shinji.probs[i])^2
            end
        end 
        for i in 1:shinji.num_arms
            shinji.gamma[i] = shinji.gamma[i] + v[i]*shinji.B/(2*shinji.gamma[i])
        end
        for i in 1:shinji.num_arms
            if i == chosen_arm
                shinji.m[i] = shinji.m[i]*(1-shinji.eta) + shinji.eta*reward
            else
                shinji.m[i] = shinji.m[i] 
            end
        end
    end
end
# Running Shinji Algorithm
function run_Shinji_algorithm(tid::Int32, monte::Int32,n_arms::Int32, n_rounds::Int32, reward_distributions::Vector{Float32}, rewards::Vector{Float32}, pulls::Matrix{Float32},name::String,randomness::Vector{Float32},importance_estimator::Matrix{Float32}, weights::Matrix{Float32}, test_estimator::Matrix{Float32})
    shinji = Shinji(n_arms)
    chosen_arm::Int32 = 0
    reward::Float32 = 0.0
    estimator = zeros(Float32,shinji.num_arms)
    for i in 1:n_rounds
        if i % 1000000 == 0
            println("Thread $tid in Monte Carlo iteration $monte inner round $i of task $name")
        end     
        update!(shinji)
        chosen_arm = select_arm(shinji)
        weights[i,:] = shinji.probs .+ 0.0
        reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
        update!!(shinji, chosen_arm, reward)        
        rewards[i] += shinji.reward[1]
        pulls[i, :] .= pulls[i, :] .+ shinji.counts
        importance_estimator[i,:] = shinji.cumu_reward .- estimator .+ 0.0
        estimator = shinji.cumu_reward .+ 0.0
        test_estimator[i,:] = shinji.cumu_reward./ i
    end
end


# # Tiancheng's FTRL
# struct Tiancheng    
#     num_arms::Int32
#     probs::Vector{Float32}
#     counts::Vector{Int32}
#     reward::Vector{Float32}
#     cumu_reward::Vector{Float32}
#     importance_estimator::Vector{Float32}
#     warmup::Vector{Float32}
#     gamma :: Vector{Float32}
#     sum_past :: Vector{Float32}
# end
# function Tiancheng(num_arms,n_rounds)
#     Tiancheng(num_arms, ones(Float32,num_arms) ./ num_arms, zeros(Int32,num_arms), zeros(Float32,1), zeros(Float32,num_arms),zeros(Float32,num_arms), ones(Float32,1)*Float32(5), ones(Float32,num_arms).* Float32(sqrt(1/log(n_rounds))), zeros(Float32,num_arms))
# end
# function select_arm(tiancheng::Tiancheng)
#     return rand(Categorical(tiancheng.probs))  # Categorical sampling
# end

# function update!(tiancheng::Tiancheng, initial::Float32)
#     tiancheng.probs .= newton!(tiancheng, initial).+0.0
# end
# function newton!(tiancheng::Tiancheng, initial::Float32)
#     temp = zeros(Float64,tiancheng.num_arms)
#     temp1 = zeros(Float64,tiancheng.num_arms)   
#     temp2 = zeros(Float32,tiancheng.num_arms)
#     if sum(tiancheng.cumu_reward)< 1e-9
#         return ones(Float32,tiancheng.num_arms)./ tiancheng.num_arms
#     end
#     @inbounds begin
#         while abs(sum(temp)-1.0) > 1e-6           
#             @simd for i in 1:tiancheng.num_arms
#                 @inbounds temp[i] = (162*log(tiancheng.num_arms)/tiancheng.gamma[i])/lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0)
#                 @inbounds temp1[i] = (162*log(tiancheng.num_arms)/tiancheng.gamma[i])/(tiancheng.gamma[i]*lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0)*(1+lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0)))
#             end
#             initial = initial - (sum(temp) - 1.0)/(sum(temp1))
#             # println(temp)
#             # println(initial)
#         end
#     end
#     @simd for i in 1:tiancheng.num_arms
#         @inbounds temp2[i] = Float32((162*log(tiancheng.num_arms)/tiancheng.gamma[i])/lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0))
#     end
#     tiancheng.warmup[1] = initial
#     return temp2 ./ sum(temp2) 
# end
# function update!!(tiancheng::Tiancheng, chosen_arm::Int32, reward::Float32, n_rounds::Int32)
#     @inbounds begin
#         tiancheng.counts[chosen_arm] += 1
#         tiancheng.reward[1] += reward
#         for i in 1:tiancheng.num_arms
#             if i == chosen_arm
#                 tiancheng.importance_estimator[i] = reward/tiancheng.probs[i] 
#             else
#                 tiancheng.importance_estimator[i] = 0.0
#             end
#         end
#         tiancheng.cumu_reward .= tiancheng.cumu_reward .+ tiancheng.importance_estimator.+0.0
#         for i in 1:tiancheng.num_arms
#             tiancheng.sum_past[i] = tiancheng.sum_past[i] + max(tiancheng.probs[i],1/n_rounds)^(-1)
#         end
#         for i in 1:tiancheng.num_arms
#             tiancheng.gamma[i] = sqrt(1/log(n_rounds))*sqrt(1+ tiancheng.sum_past[i])
#         end       
#     end
# end
# # Running Tiancheng Algorithm
# function run_Tiancheng_algorithm(tid::Int32, monte::Int32,n_arms::Int32, n_rounds::Int32, reward_distributions::Vector{Float32}, rewards::Vector{Float32}, pulls::Matrix{Float32},name::String,randomness::Vector{Float32},importance_estimator::Matrix{Float32}, weights::Matrix{Float32}, test_estimator::Matrix{Float32})
#     tiancheng = Tiancheng(n_arms,n_rounds)
#     chosen_arm::Int32 = 0
#     reward::Float32 = 0.0
#     estimator = zeros(Float32,tiancheng.num_arms)
#     for i in 1:n_rounds
#         if i % 1000000 == 0
#             println("Thread $tid in Monte Carlo iteration $monte inner round $i of task $name")
#         end     
#         update!(tiancheng,tiancheng.warmup[1])
#         chosen_arm = select_arm(tiancheng)
#         weights[i,:] = tiancheng.probs .+ 0.0
#         reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
#         update!!(tiancheng, chosen_arm, reward, n_rounds)
#         rewards[i] += tiancheng.reward[1]
#         pulls[i, :] .= pulls[i, :] .+ tiancheng.counts
#         importance_estimator[i,:] = tiancheng.cumu_reward .- estimator .+ 0.0
#         estimator = tiancheng.cumu_reward .+ 0.0
#         test_estimator[i,:] = tiancheng.cumu_reward./ i
#     end
# end

# # Broad's FTRL
# struct Broad   
#     num_arms::Int32
#     probs::Vector{Float32}
#     counts::Vector{Int32}
#     reward::Vector{Float32}
#     cumu_reward::Vector{Float32}
#     importance_estimator::Vector{Float32}
#     warmup::Vector{Float32}
# end
# function Broad(num_arms)
#     Broad(num_arms, ones(Float32,num_arms) ./ num_arms, zeros(Int32,num_arms), zeros(Float32,1), zeros(Float32,num_arms),zeros(Float32,num_arms), ones(Float32,1)*Float32(1.0))
# end
# function select_arm(broad::Broad)
#     return rand(Categorical(broad.probs))  # Categorical sampling
# end

# function update!(broad::Broad, initial::Float32, eta::Float32)
#     broad.probs .= newton!(broad, initial, eta).+0.0
# end
# function newton!(broad::Broad, initial::Float32, eta::Float32)
#     temp = zeros(Float64,tiancheng.num_arms)
#     temp1 = zeros(Float64,tiancheng.num_arms)   
#     temp2 = zeros(Float32,tiancheng.num_arms)
#     if sum(tiancheng.cumu_reward)< 1e-9
#         return ones(Float32,tiancheng.num_arms)./ tiancheng.num_arms
#     end
#     @inbounds begin
#         while abs(sum(temp)-1.0) > 1e-6           
#             @simd for i in 1:tiancheng.num_arms
#                 @inbounds temp[i] = (162*log(tiancheng.num_arms)/tiancheng.gamma[i])/lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0)
#                 @inbounds temp1[i] = (162*log(tiancheng.num_arms)/tiancheng.gamma[i])/(tiancheng.gamma[i]*lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0)*(1+lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0)))
#             end
#             initial = initial - (sum(temp) - 1.0)/(sum(temp1))
#         end
#     end
#     @simd for i in 1:tiancheng.num_arms
#         @inbounds temp2[i] = Float32((162*log(tiancheng.num_arms)/tiancheng.gamma[i])/lambertw((162*log(tiancheng.num_arms)/tiancheng.gamma[i])*exp((tiancheng.cumu_reward[i]-initial)/tiancheng.gamma[i]),0))
#     end
#     tiancheng.warmup[1] = initial
#     return temp2 ./ sum(temp2) 
# end
# function update!!(broad ::Broad, chosen_arm::Int32, reward::Float32, n_rounds::Int32)
#     @inbounds begin
#         broad.counts[chosen_arm] += 1
#         broad.reward[1] += reward
#         for i in 1:tiancheng.num_arms
#             if i == chosen_arm
#                 broad.importance_estimator[i] = reward/tiancheng.probs[i] 
#             else
#                 broad.importance_estimator[i] = 0.0
#             end
#         end
#         broad.cumu_reward .= broad.cumu_reward .+ broad.importance_estimator.+0.0  
#     end
# end
# # Running Broad Algorithm
# function run_Broad_algorithm(tid::Int32, monte::Int32,n_arms::Int32, n_rounds::Int32, reward_distributions::Vector{Float32}, rewards::Vector{Float32}, pulls::Matrix{Float32},name::String,randomness::Vector{Float32},importance_estimator::Matrix{Float32}, weights::Matrix{Float32}, test_estimator::Matrix{Float32})
#     broad = Broad(n_arms)
#     chosen_arm::Int32 = 0
#     reward::Float32 = 0.0
#     estimator = zeros(Float32,tiancheng.num_arms)
#     init :: Float32 = 1.0 
#     for i in 1:n_rounds
#         if i % 1000000 == 0
#             println("Thread $tid in Monte Carlo iteration $monte inner round $i of task $name")
#         end   
#         if init > 0.0
#             broad.probs .= ones(Float32,n_arms) ./ n_arms
#             init = -1.0
#         end  
       
#         chosen_arm = select_arm(tiancheng)
#         weights[i,:] = tiancheng.probs .+ 0.0
#         reward = (randomness[i] < reward_distributions[chosen_arm]) * Float32(2.0)  # Optimize random binomial
#         update!!(tiancheng, chosen_arm, reward, n_rounds)
#         rewards[i] += tiancheng.reward[1]
#         pulls[i, :] .= pulls[i, :] .+ tiancheng.counts
#         importance_estimator[i,:] = tiancheng.cumu_reward .- estimator .+ 0.0
#         estimator = tiancheng.cumu_reward .+ 0.0
#         test_estimator[i,:] = tiancheng.cumu_reward./ i
#     end
# end



function main(n_rounds::Int32,n_montecarlo::Int32, n_arms::Int32)
    # Main code for running the algorithms and plotting the results

    reward_distributions = ones(Float32,n_arms)*Float32(0.2) 
    reward_distributions[1] = 0.0


   # 0.01 4 


    # Use Float32 for larger data arrays to optimize memory usage
    cumulative_reward = zeros(Float32, n_rounds)
    average_pulls = zeros(Float32, n_rounds, n_arms)

    cumulative_reward_1 = zeros(Float32, n_rounds)
    average_pulls_1 = zeros(Float32, n_rounds, n_arms)

    cumulative_reward_2 = zeros(Float32, n_rounds)
    average_pulls_2 = zeros(Float32, n_rounds, n_arms)

    cumulative_reward_3 = zeros(Float32, n_rounds)
    average_pulls_3 = zeros(Float32, n_rounds, n_arms)

    cumulative_reward_4 = zeros(Float32, n_rounds)
    average_pulls_4 = zeros(Float32, n_rounds, n_arms)

    cumulative_reward_5 = zeros(Float32, n_rounds)
    average_pulls_5 = zeros(Float32, n_rounds, n_arms)

    cumulative_reward_6 = zeros(Float32, n_rounds)
    average_pulls_6 = zeros(Float32, n_rounds, n_arms)

    # cumulative_reward_7 = zeros(Float32, n_rounds)
    # average_pulls_7 = zeros(Float32, n_rounds, n_arms)

    importance_estimator_1 = zeros(Float32, n_rounds, n_arms)
    importance_estimator_2 = zeros(Float32, n_rounds, n_arms)
    importance_estimator_3 = zeros(Float32, n_rounds, n_arms)
    importance_estimator_4 = zeros(Float32, n_rounds, n_arms)
    importance_estimator_5 = zeros(Float32, n_rounds, n_arms)
    importance_estimator_6 = zeros(Float32, n_rounds, n_arms)
    # importance_estimator_7 = zeros(Float32, n_rounds, n_arms)

    weights_1 = zeros(Float32, n_rounds, n_arms)
    weights_2 = zeros(Float32, n_rounds, n_arms)
    weights_3 = zeros(Float32, n_rounds, n_arms)
    weights_4 = zeros(Float32, n_rounds, n_arms)
    weights_5 = zeros(Float32, n_rounds, n_arms)
    weights_6 = zeros(Float32, n_rounds, n_arms)
    # weights_7 = zeros(Float32, n_rounds, n_arms)

    test_estimator_1 = zeros(Float32, n_rounds, n_arms)
    test_estimator_2 = zeros(Float32, n_rounds, n_arms)
    test_estimator_3 = zeros(Float32, n_rounds, n_arms)
    test_estimator_4 = zeros(Float32, n_rounds, n_arms)
    test_estimator_5 = zeros(Float32, n_rounds, n_arms)
    test_estimator_6 = zeros(Float32, n_rounds, n_arms)
    # test_estimator_7 = zeros(Float32, n_rounds, n_arms)



    # Create a ReentrantLock for atomic updates
    lock = ReentrantLock()
    lock_1 = ReentrantLock()
    lock_2 = ReentrantLock()
    lock_3 = ReentrantLock()
    lock_4 = ReentrantLock()
    lock_5 = ReentrantLock()
    lock_6 = ReentrantLock()
    lock_7 = ReentrantLock()
    lock_8 = ReentrantLock()
    lock_9 = ReentrantLock()
    lock_10 = ReentrantLock()
    lock_11 = ReentrantLock()
    lock_12 = ReentrantLock()
    lock_13 = ReentrantLock()
    lock_14 = ReentrantLock()
    lock_15 = ReentrantLock()
    lock_16 = ReentrantLock()
    lock_17 = ReentrantLock()
    lock_18 = ReentrantLock()
    lock_19 = ReentrantLock()
    lock_20 = ReentrantLock()
    lock_21 = ReentrantLock()
    lock_22 = ReentrantLock()
    lock_23 = ReentrantLock()
    lock_24 = ReentrantLock()
    lock_25 = ReentrantLock()
    lock_26 = ReentrantLock()
    lock_27 = ReentrantLock()
    lock_28 = ReentrantLock()
    lock_29 = ReentrantLock()
    lock_30 = ReentrantLock()
    lock_31 = ReentrantLock()
    # lock_32 = ReentrantLock()
    # lock_33 = ReentrantLock()
    # lock_34 = ReentrantLock()
    # lock_35 = ReentrantLock()
    # lock_36 = ReentrantLock()


    # Start timing
    start_time = now()  # Capture the start time
    name=["UCB","EXP3_C","EXP3_V_FTRL","EXP3_V_OMD", "EXP3_++","Tsallisinf", "Shinji"]
    # Multithreaded Monte Carlo simulations
    @threads for i in 1:n_montecarlo
        tid = threadid()
        println("Thread $tid starting Monte Carlo iteration $i")
        local_cumulative_reward = zeros(Float32,n_rounds)
        local_average_pulls = zeros(Float32,n_rounds, n_arms)

        local_cumulative_reward_1 = zeros(Float32,n_rounds)
        local_average_pulls_1 = zeros(Float32,n_rounds, n_arms)

        local_cumulative_reward_2 = zeros(Float32,n_rounds)
        local_average_pulls_2 = zeros(Float32,n_rounds, n_arms)

        local_cumulative_reward_3 = zeros(Float32,n_rounds)
        local_average_pulls_3 = zeros(Float32,n_rounds, n_arms)

        local_cumulative_reward_4 = zeros(Float32,n_rounds)
        local_average_pulls_4 = zeros(Float32,n_rounds, n_arms)

        local_cumulative_reward_5 = zeros(Float32,n_rounds)
        local_average_pulls_5 = zeros(Float32,n_rounds, n_arms)

        local_cumulative_reward_6 = zeros(Float32,n_rounds)
        local_average_pulls_6 = zeros(Float32,n_rounds, n_arms)

        # local_cumulative_reward_7 = zeros(Float32,n_rounds)
        # local_average_pulls_7 = zeros(Float32,n_rounds, n_arms)

        local_importance_estimator_1 = zeros(Float32,n_rounds, n_arms)
        local_importance_estimator_2 = zeros(Float32,n_rounds, n_arms)
        local_importance_estimator_3 = zeros(Float32,n_rounds, n_arms)
        local_importance_estimator_4 = zeros(Float32,n_rounds, n_arms)
        local_importance_estimator_5 = zeros(Float32,n_rounds, n_arms)
        local_importance_estimator_6 = zeros(Float32,n_rounds, n_arms)
        # local_importance_estimator_7 = zeros(Float32,n_rounds, n_arms)

        local_weights_1 = zeros(Float32,n_rounds, n_arms)
        local_weights_2 = zeros(Float32,n_rounds, n_arms)
        local_weights_3 = zeros(Float32,n_rounds, n_arms)
        local_weights_4 = zeros(Float32,n_rounds, n_arms)
        local_weights_5 = zeros(Float32,n_rounds, n_arms)
        local_weights_6 = zeros(Float32,n_rounds, n_arms)
        # local_weights_7 = zeros(Float32,n_rounds, n_arms)

        local_test_estimator_1 = zeros(Float32,n_rounds, n_arms)
        local_test_estimator_2 = zeros(Float32,n_rounds, n_arms)
        local_test_estimator_3 = zeros(Float32,n_rounds, n_arms)
        local_test_estimator_4 = zeros(Float32,n_rounds, n_arms)
        local_test_estimator_5 = zeros(Float32,n_rounds, n_arms)
        local_test_estimator_6 = zeros(Float32,n_rounds, n_arms)
        # local_test_estimator_7 = zeros(Float32,n_rounds, n_arms)

        # Set the random seed for each thread
        Random.seed!(tid)
        # Create Randomness for each inner round
        randomness = rand(Float32, n_rounds)
        # Run the algorithms and accumulate results locally
        run_ucb_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, local_cumulative_reward, local_average_pulls,name[1],randomness)
        run_exp3_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, Int32(0), local_cumulative_reward_1, local_average_pulls_1,name[2],randomness, local_importance_estimator_1, local_weights_1, local_test_estimator_1)
        run_exp3_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, Int32(1), local_cumulative_reward_2, local_average_pulls_2,name[3],randomness, local_importance_estimator_2, local_weights_2, local_test_estimator_2)
        run_exp3_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, Int32(2), local_cumulative_reward_3, local_average_pulls_3,name[4],randomness, local_importance_estimator_3, local_weights_3, local_test_estimator_3)
        run_exp3plus_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, local_cumulative_reward_4, local_average_pulls_4,name[5],randomness, local_importance_estimator_4, local_weights_4, local_test_estimator_4)
        run_Tsallisinf_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, local_cumulative_reward_5, local_average_pulls_5,name[6],randomness, local_importance_estimator_5, local_weights_5, local_test_estimator_5)
        run_Shinji_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, local_cumulative_reward_6, local_average_pulls_6,name[7],randomness, local_importance_estimator_6, local_weights_6, local_test_estimator_6)
        # run_Broad_algorithm(Int32(tid),Int32(i),n_arms, n_rounds, reward_distributions, local_cumulative_reward_7, local_average_pulls_7,name[8],randomness, local_importance_estimator_7, local_weights_7, local_test_estimator_7)
        # Use a lock to update the shared arrays
        @lock lock cumulative_reward .+= local_cumulative_reward
        @lock lock_1 average_pulls .+= local_average_pulls
        @lock lock_2 cumulative_reward_1 .+= local_cumulative_reward_1
        @lock lock_3 average_pulls_1 .+= local_average_pulls_1
        @lock lock_4 cumulative_reward_2 .+= local_cumulative_reward_2
        @lock lock_5 average_pulls_2 .+= local_average_pulls_2
        @lock lock_6 cumulative_reward_3 .+= local_cumulative_reward_3
        @lock lock_7 average_pulls_3 .+= local_average_pulls_3
        @lock lock_8 cumulative_reward_4 .+= local_cumulative_reward_4
        @lock lock_9 average_pulls_4 .+= local_average_pulls_4
        @lock lock_10 importance_estimator_1 .+= local_importance_estimator_1
        @lock lock_11 importance_estimator_2 .+= local_importance_estimator_2
        @lock lock_12 importance_estimator_3 .+= local_importance_estimator_3
        @lock lock_13 importance_estimator_4 .+= local_importance_estimator_4
        @lock lock_14 weights_1 .+= local_weights_1
        @lock lock_15 weights_2 .+= local_weights_2
        @lock lock_16 weights_3 .+= local_weights_3
        @lock lock_17 weights_4 .+= local_weights_4
        @lock lock_18 test_estimator_1 .+= local_test_estimator_1
        @lock lock_19 test_estimator_2 .+= local_test_estimator_2
        @lock lock_20 test_estimator_3 .+= local_test_estimator_3
        @lock lock_21 test_estimator_4 .+= local_test_estimator_4
        @lock lock_22 cumulative_reward_5 .+= local_cumulative_reward_5
        @lock lock_23 average_pulls_5 .+= local_average_pulls_5
        @lock lock_24 importance_estimator_5 .+= local_importance_estimator_5
        @lock lock_25 weights_5 .+= local_weights_5
        @lock lock_26 test_estimator_5 .+= local_test_estimator_5
        @lock lock_27 cumulative_reward_6 .+= local_cumulative_reward_6
        @lock lock_28 average_pulls_6 .+= local_average_pulls_6
        @lock lock_29 importance_estimator_6 .+= local_importance_estimator_6
        @lock lock_30 weights_6 .+= local_weights_6
        @lock lock_31 test_estimator_6 .+= local_test_estimator_6
        # @lock lock_32 cumulative_reward_7 .+= local_cumulative_reward_7
        # @lock lock_33 average_pulls_7 .+= local_average_pulls_7
        # @lock lock_34 importance_estimator_7 .+= local_importance_estimator_7
        # @lock lock_35 weights_7 .+= local_weights_7
        # @lock lock_36 test_estimator_7 .+= local_test_estimator_7
    end

    # End timing and print the total elapsed time
    elapsed_time = now() - start_time
    println("Total elapsed time: ", elapsed_time)


    # Averaging the results
    cumulative_reward ./= n_montecarlo
    average_pulls ./= n_montecarlo

    cumulative_reward_1 ./= n_montecarlo
    average_pulls_1 ./= n_montecarlo

    cumulative_reward_2 ./= n_montecarlo
    average_pulls_2 ./= n_montecarlo

    cumulative_reward_3 ./= n_montecarlo
    average_pulls_3 ./= n_montecarlo

    cumulative_reward_4 ./= n_montecarlo
    average_pulls_4 ./= n_montecarlo

    cumulative_reward_5 ./= n_montecarlo
    average_pulls_5 ./= n_montecarlo

    cumulative_reward_6 ./= n_montecarlo
    average_pulls_6 ./= n_montecarlo

    # cumulative_reward_7 ./= n_montecarlo
    # average_pulls_7 ./= n_montecarlo

    importance_estimator_1 ./= n_montecarlo
    importance_estimator_2 ./= n_montecarlo
    importance_estimator_3 ./= n_montecarlo
    importance_estimator_4 ./= n_montecarlo
    importance_estimator_5 ./= n_montecarlo
    importance_estimator_6 ./= n_montecarlo
    # importance_estimator_7 ./= n_montecarlo

    weights_1 ./= n_montecarlo
    weights_2 ./= n_montecarlo
    weights_3 ./= n_montecarlo
    weights_4 ./= n_montecarlo
    weights_5 ./= n_montecarlo
    weights_6 ./= n_montecarlo
    # weights_7 ./= n_montecarlo

    test_estimator_1 ./= n_montecarlo
    test_estimator_2 ./= n_montecarlo
    test_estimator_3 ./= n_montecarlo
    test_estimator_4 ./= n_montecarlo
    test_estimator_5 ./= n_montecarlo
    test_estimator_6 ./= n_montecarlo
    # test_estimator_7 ./= n_montecarlo

    pseudo_regret = cumulative_reward - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    pseudo_regret_1 = cumulative_reward_1 - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    pseudo_regret_2 = cumulative_reward_2 - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    pseudo_regret_3 = cumulative_reward_3 - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    pseudo_regret_4 = cumulative_reward_4 - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    pseudo_regret_5 = cumulative_reward_5 - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    pseudo_regret_6 = cumulative_reward_6 - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    # pseudo_regret_7 = cumulative_reward_7 - (1:n_rounds) .* (minimum(reward_distributions) * Float32(2.0))
    gap1 = sort(reward_distributions) 
    gap = gap1[2] - gap1[1]
    # Plotting the pseudo-regret and save to bin file
    plot(1:n_rounds, pseudo_regret, label="UCB", xscale=:log10, xlabel="Rounds", ylabel="Pseudo-regret", title="Pseudo-regret of the UCB/EXP3/Tsallisinf/log-barrier algorithm",legend=:left)
    plot!(1:n_rounds, pseudo_regret_1, label="EXP3_C")
    plot!(1:n_rounds, pseudo_regret_2, label="EXP3_V_FTRL")
    plot!(1:n_rounds, pseudo_regret_3, label="EXP3_V_OMD")
    plot!(1:n_rounds, pseudo_regret_4, label="EXP3_++")
    plot!(1:n_rounds, pseudo_regret_5, label="Tsallisinf")
    plot!(1:n_rounds, pseudo_regret_6, label="Shinji")
    # plot!(1:n_rounds, pseudo_regret_7, label="Broad")
    savefig("regret_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).png") 
    # Save to bin file with name psudo_regret_name[i]_n_rounds_n_monte_n_arms.bin 
    open("psudo_regret_UCB_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
        serialize(io, pseudo_regret)
    end
    open("psudo_regret_EXP3_C_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
        serialize(io, pseudo_regret_1)
    end
    open("psudo_regret_EXP3_V_FTRL_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
        serialize(io, pseudo_regret_2)
    end
    open("psudo_regret_EXP3_V_OMD_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
        serialize(io, pseudo_regret_3)
    end
    open("psudo_regret_EXP3_++_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
        serialize(io, pseudo_regret_4)
    end
    open("psudo_regret_Tsallisinf_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
        serialize(io, pseudo_regret_5)
    end
    open("psudo_regret_Shinji_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
        serialize(io, pseudo_regret_6)
    end
    # open("psudo_regret_Broad_$(n_rounds)_$(n_montecarlo)_$(n_arms).bin", "w") do io
    #     serialize(io, pseudo_regret_7)
    # end
    

    plot(1:n_rounds, pseudo_regret, label="UCB", xscale=:log10, xlabel="Rounds", ylabel="Pseudo-regret", title="Pseudo-regret of the UCB/EXP3/Tsallisinf/log-barrier algorithm",legend=:left)
    plot!(1:n_rounds, pseudo_regret_2, label="EXP3_V_FTRL")
    plot!(1:n_rounds, pseudo_regret_3, label="EXP3_V_OMD")
    plot!(1:n_rounds, pseudo_regret_4, label="EXP3_++")
    plot!(1:n_rounds, pseudo_regret_5, label="Tsallisinf")
    plot!(1:n_rounds, pseudo_regret_6, label="Shinji")
    # plot!(1:n_rounds, pseudo_regret_7, label="Broad")
    savefig("regret_$(n_rounds)_1_$(n_montecarlo)_$(n_arms)_$(gap).png") 
    # Plotting the pulls for each arm
    for i in 1:n_arms
        plot(1:n_rounds, average_pulls[:, i], label="UCB", xscale=:log10, xlabel="Rounds", ylabel="Number of pulls", title="Number of pulls for arm $(i)",legend=:left)
        plot!(1:n_rounds, average_pulls_1[:, i], label="EXP3_C")
        plot!(1:n_rounds, average_pulls_2[:, i], label="EXP3_V_FTRL")
        plot!(1:n_rounds, average_pulls_3[:, i], label="EXP3_V_OMD")
        plot!(1:n_rounds, average_pulls_4[:, i], label="EXP3_++")
        plot!(1:n_rounds, average_pulls_5[:, i], label="Tsallisinf")
        plot!(1:n_rounds, average_pulls_6[:, i], label="Shinji")
        # plot!(1:n_rounds, average_pulls_7[:, i], label="Broad")
        savefig("arm_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).png") 
        # Save to bin file with name pulls_name[i]_n_rounds_n_monte_n_arms.bin 
        open("pulls_UCB_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, average_pulls[:, i])
        end
        open("pulls_EXP3_C_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, average_pulls_1[:, i])
        end
        open("pulls_EXP3_V_FTRL_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, average_pulls_2[:, i])
        end
        open("pulls_EXP3_V_OMD_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, average_pulls_3[:, i])
        end
        open("pulls_EXP3_++_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, average_pulls_4[:, i])
        end
        open("pulls_Tsallisinf_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, average_pulls_5[:, i])
        end
        open("pulls_Shinji_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, average_pulls_6[:, i])
        end
        # open("pulls_Broad_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms).bin", "w") do io
        #     serialize(io, average_pulls_7[:, i])
        # end
    end
    # Plotting the importance estimator for each arm
    for i in 1:n_arms
        plot(1:n_rounds, importance_estimator_1[:, i], label="EXP3_C", xscale=:log10,xlabel="Rounds", ylabel="Importance Estimator", title="Importance Estimator for arm $(i)",legend=:left)
        plot!(1:n_rounds, importance_estimator_2[:, i], label="EXP3_V_FTRL")
        plot!(1:n_rounds, importance_estimator_3[:, i], label="EXP3_V_OMD")
        plot!(1:n_rounds, importance_estimator_4[:, i], label="EXP3_++")
        plot!(1:n_rounds, importance_estimator_5[:, i], label="Tsallisinf")
        plot!(1:n_rounds, importance_estimator_6[:, i], label="Shinji")
        # plot!(1:n_rounds, importance_estimator_7[:, i], label="Broad")
        savefig("importance_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).png") 
        # Save to bin file with name importance_name[i]_n_rounds_n_monte_n_arms.bin
        open("importance_estimator_EXP3_C_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, importance_estimator_1[:, i])
        end
        open("importance_estimator_EXP3_V_FTRL_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, importance_estimator_2[:, i])
        end
        open("importance_estimator_EXP3_V_OMD_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, importance_estimator_3[:, i])
        end
        open("importance_estimator_EXP3_++_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, importance_estimator_4[:, i])
        end
        open("importance_estimator_Tsallisinf_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, importance_estimator_5[:, i])
        end
        open("importance_estimator_Shinji_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, importance_estimator_6[:, i])
        end
        # open("importance_estimator_Broad_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms).bin", "w") do io
        #     serialize(io, importance_estimator_7[:, i])
        # end
    end
    # Plotting the weights for each arm
    for i in 1:n_arms
        plot(1:n_rounds, weights_1[:, i], label="EXP3_C", xscale=:log10, xlabel="Rounds", ylabel="Weights", title="Weights for arm $(i)",legend=:left)
        plot!(1:n_rounds, weights_2[:, i], label="EXP3_V_FTRL")
        plot!(1:n_rounds, weights_3[:, i], label="EXP3_V_OMD")
        plot!(1:n_rounds, weights_4[:, i], label="EXP3_++")
        plot!(1:n_rounds, weights_5[:, i], label="Tsallisinf")
        plot!(1:n_rounds, weights_6[:, i], label="Shinji")
        # plot!(1:n_rounds, weights_7[:, i], label="Broad")
        savefig("weights_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).png") 
        # Save to bin file with name weights_name[i]_n_rounds_n_monte_n_arms.bin 
        open("weights_EXP3_C_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, weights_1[:, i])
        end
        open("weights_EXP3_V_FTRL_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, weights_2[:, i])
        end
        open("weights_EXP3_V_OMD_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, weights_3[:, i])
        end
        open("weights_EXP3_++_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, weights_4[:, i])
        end
        open("weights_Tsallisinf_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, weights_5[:, i])
        end
        open("weights_Shinji_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, weights_6[:, i])
        end
        # open("weights_Broad_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms).bin", "w") do io
        #     serialize(io, weights_7[:, i])
        # end
    end

    # Plotting the test estimator for each arm
    for i in 1:n_arms
        plot(1:n_rounds, test_estimator_1[:, i], label="EXP3_C", xscale=:log10, xlabel="Rounds", ylabel="Test Estimator", title="Test Estimator for arm $(i)",legend=:left)
        plot!(1:n_rounds, test_estimator_2[:, i], label="EXP3_V_FTRL")
        plot!(1:n_rounds, test_estimator_3[:, i], label="EXP3_V_OMD")
        plot!(1:n_rounds, test_estimator_4[:, i], label="EXP3_++")
        plot!(1:n_rounds, test_estimator_5[:, i], label="Tsallisinf")
        plot!(1:n_rounds, test_estimator_6[:, i], label="Shinji")
        # plot!(1:n_rounds, test_estimator_7[:, i], label="Broad")
        savefig("test_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).png") 
        # Save to bin file with name estimator_name[i]_n_rounds_n_monte_n_arms.bin 
        open("test_estimator_EXP3_C_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, test_estimator_1[:, i])
        end
        open("test_estimator_EXP3_V_FTRL_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, test_estimator_2[:, i])
        end
        open("test_estimator_EXP3_V_OMD_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, test_estimator_3[:, i])
        end
        open("test_estimator_EXP3_++_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, test_estimator_4[:, i])
        end
        open("test_estimator_Tsallisinf_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, test_estimator_5[:, i])
        end
        open("test_estimator_Shinji_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms)_$(gap).bin", "w") do io
            serialize(io, test_estimator_6[:, i])
        end
        # open("test_estimator_Broad_$(i)_$(n_rounds)_$(n_montecarlo)_$(n_arms).bin", "w") do io
        #     serialize(io, test_estimator_7[:, i])
        # end
    end

end
n_rounds = parse(Int32,ARGS[1])
n_montecarlo = parse(Int32,ARGS[2])
n_arms = parse(Int32,ARGS[3])
@time main(n_rounds,n_montecarlo,n_arms)
