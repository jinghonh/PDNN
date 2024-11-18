function [inner_vert] = reachset(A,B,T,N,lb,ub)
    % format long;
    point_num = 600;%ex03 need a square number
    step = T/N;
    dim = size(A,1);
    %% calculate X_N
    P = [];
    for k = 0:N-1
        Q = eye(dim) + step * A;
        multi_Q = Q^(N-1-k);
        temp = step * multi_Q * B;
        P = [P temp];
    end 
    % save model_dim10_001 P lb ub
    %% solve problem
    example_num  = 2;
    saveu_flag = 1;
    [inner_vert,~,~] = alg_dir(P,lb,ub,example_num,N,point_num,saveu_flag);
end

