function [inner_vert,LP_num,opt_u] = alg_dir(P,lb,ub,example_num,N,point_num,saveu_flag)
% dir method 
% provide the internal and external approximation for reachable sets
dim = size(P,1);
inner_vert = [];
LP_num = 0;
opt_u = [];
%% generate directions
dir = gene_dir(point_num,dim);
point_num_real = size(dir,1);
%% solve prob
for i = 1:point_num_real
    obj_P = -dir(i,:) * P;
    clear prob;
    prob.c = obj_P';
    if example_num == 1 || example_num == 400 || example_num == 4
        prob.a = sparse(1,N);
        prob.blx = lb;
        prob.bux = ub;
    elseif example_num == 2
        prob.qcsubk = reshape([1:N;1:N],1,2*N)';
        prob.qcsubi = (1:2*N)';
        prob.qcsubj = (1:2*N)';
        prob.qcval  = 2 * ones(1,2*N)';
        prob.a = sparse(N,2*N);
        prob.buc = ones(N,1);
    else
        temp = repmat(1:N,dim,1);
        prob.qcsubk = reshape(temp,1,dim*N)';
        prob.qcsubi = (1:dim*N)';
        prob.qcsubj = (1:dim*N)';
        prob.qcval  = 2 * ones(1,dim*N)';
        prob.a = sparse(N,dim*N);
        prob.buc = ones(N,1);
    end
    [~,res] = mosekopt('minimize echo(0)',prob); 
    x = res.sol.itr.xx;
    LP_num = LP_num + 1;
    if saveu_flag
        opt_u = [opt_u;x'];
    end
    boundary = P * x;
    inner_vert = [inner_vert;boundary'];
end
LP_num;

