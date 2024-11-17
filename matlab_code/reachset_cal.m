%% 
close all;
clear;
clc;
% format long;
%%
example_num  = 2;%nonautonomous is 400 dim>3 use no.4
point_num = 600;%ex03 need a square number
step = 0.4;
plot_flag = 1;
saveu_flag = 1;
%% problem initialization 
if example_num == 1   
    A = [0 1;0 0];
    B = [0 1]';
    T = 1;
    N = T / step;
    lb = zeros(N,1);
    ub = ones(N,1);
elseif example_num == 2
    A = [0 1;-2 -3];
    B = eye(2);
    T = 2;
    N = T / step;
    lb = [];
    ub = [];
elseif example_num == 3
    A = [-1 1 0;
        0 -1 0;
        0 0 -2];
    B = diag([0 2 1]);
    T = 1;
    N = T / step;
    lb = [];
    ub = [];
elseif example_num == 400
    A = ones(2);
    T = 4;
    N = T / step;
    lb = -1 * ones(N,1);
    ub = ones(N,1);
elseif example_num == 4
    % dim3
    A_gene = [-1 0 1;...
               0 -2 0;...
               0 1 -1];
    B_gene = [0 1 1]';
%     % dim4
%     A_gene = [-1 0 1 0;...
%                0 -2 0 0;...
%                0 1 -1 0;...
%                0 0 0 -3];
%     B_gene = [0 1 1 2]';
    % dim5
%     A_gene = [-1 0 1 0 0;...
%            0 -2 0 0 0;...
%            0 1 -1 0 -1;...
%            0 0 0 -3 1;...
%            0 0 0 0 -1];
%     B_gene = [0 1 1 2 3]';
%     % dim6
%     A_gene = [-1 0 1 0 0 0;...
%            0 -2 0 0 0 0;...
%            0 1 -1 0 -1 0;...
%            0 0 0 -3 1 0;...
%            0 0 0 0 -1 0;...
%            0 0 0 0 -1 -1];
%     B_gene = [0 1 1 2 1 1]';
%     % dim7
%     A_gene = [-1 0 1 0 0 0 0;...
%            0 -2 0 0 0 0 0;...
%            0 1 -1 0 -1 0 0;...
%            0 0 0 -3 1 0 0;...
%            0 0 0 0 -1 0 0;...
%            0 0 0 0 -1 -1 0;...
%            0 0 0 0 -1 0 -1];
%     B_gene = [0 1 1 2 1 1 2]';
%     % dim8
%     A_gene = [-1 0 1 0 0 0 0 0;...
%            0 -2 0 0 0 0 0 0;...
%            0 1 -1 0 -1 0 0 0;...
%            0 0 0 -3 1 0 0 0;...
%            0 0 0 0 -1 0 0 0;...
%            0 0 0 0 -1 -1 0 0;...
%            0 0 0 0 -1 0 -1 0;...
%            0 0 0 0 0 0 -1 -1];
%     B_gene = [0 1 1 2 1 1 2 1]';
%     % dim9
%     A_gene = [-1 0 1 0 0 0 0 0 0;...
%            0 -2 0 0 0 0 0 0 0;...
%            0 1 -1 0 -1 0 0 0 0;...
%            0 0 0 -3 1 0 0 0 0;...
%            0 0 0 0 -1 0 0 0 0;...
%            0 0 0 0 -1 -1 0 0 0;...
%            0 0 0 0 -1 0 -1 0 0;...
%            0 0 0 0 0 0 -1 -2 0;...
%            0 0 0 0 0 0 0 -1 -3];
%     B_gene = [0 1 1 2 1 1 2 1 1]';
%     % dim10
%     A_gene = [-1 0 1 0 0 0 0 0 0 0;...
%            0 -2 0 0 0 0 0 0 0 0;...
%            0 1 -1 0 -1 0 0 0 0 0;...
%            0 0 0 -3 1 0 0 0 0 0;...
%            0 0 0 0 -1 0 0 0 0 0;...
%            0 0 0 0 -1 -1 0 0 0 0;...
%            0 0 0 0 -1 0 -1 0 0 0;...
%            0 0 0 0 0 0 -1 -2 0 0;...
%            0 0 0 0 0 0 0 -1 -3 0;...
%            0 0 0 0 0 0 0 -1 -3 -2];
%     B_gene = [0 1 1 2 1 1 2 1 1 3]';
    

    A = A_gene;
    B = B_gene;
    T = 1;
    N = T / step;
    lb = -ones(N,1);
    ub = ones(N,1);
end
dim = size(A,1);
%% calculate X_N
P = [];
for k = 0:N-1
    if example_num < 100
        Q = eye(dim) + step * A;
        multi_Q = Q^(N-1-k);
    else
        multi_Q = eye(dim);
        for j = k + 1:N - 1
            t = j * step;
            if example_num == 400
                Q = eye(dim) + step * [exp(-t)*sin(t)-2 t*exp(-2*t); 
                    -exp(-t) 2*exp(-2*t)*cos(t)-1];
            end
            multi_Q = multi_Q * Q;
        end
        t = k * step;
        if example_num == 400
            B = [1 sin(t)]';
        end
    end
    temp = step * multi_Q * B;
    P = [P temp];
end 
% save model_dim10_001 P lb ub
%% solve problem
[inner_vert,LP_num,opt_u] = alg_dir(P,lb,ub,example_num,N,point_num,saveu_flag);
% save opt_control opt_u
%% plot result
if plot_flag
	if dim < 3
        plot_result(inner_vert,'r','-','y',1,1,1);
        hold on;
        plot(inner_vert(:,1),inner_vert(:,2),'k.');
        hold on;
    else
       plot_result(inner_vert,'none','k','y',1,1,1);
       hold on;
       plot3(inner_vert(:,1),inner_vert(:,2),inner_vert(:,3),'k.');
       hold on;
    end
    xlabel('x_1');
    ylabel('x_2');
    if dim > 2
        zlabel('x_3');
    end
end

