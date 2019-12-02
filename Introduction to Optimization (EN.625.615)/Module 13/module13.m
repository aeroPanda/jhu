%% Module 13
%% Question 1
syms x
f = exp(sqrt(2*x)) + 2*x;
fp = diff(f,'x');
fpp = diff(fp,'x');
b = solve(fpp==0);
disp(['b = ' num2str(eval(b))]);
%%
hF = figure;
hA = axes(hF,'NextPlot','add');
hA.YLim = [-10 10];
hA.XGrid = 'on';
hA.YGrid = 'on';
fplot(f,[0 1],'blue')
fplot(fp,[0 1],'green')
fplot(fpp,[0 1],'red')
scatter(b,subs(fpp,x,b),'o','red')

%% Question 2
syms x y z
f_xyz = x^5 + y^4 + z^3 + x*y^2 + y*z + 1;
h = hessian(f_xyz);
h_1 = eval( subs(h,{x,y,z},{1 -1 1}) );
disp(eig(h_1))
% positive definite

%% Question 3
% f(x,yg*)
%% Question 4
% f(x,yf*)

%% Question 5
% second

%% Question 6
% second

%% Question 7
% f(xs,ys) <= f(x,ys)

%% Question 8
% f((xs,y) <= f(xs,ys)

%% Question 9
syms x y
f_xy = x^2 -3*y^2;
f_xy_xp = diff(f_xy,x);
f_xy_xpp = diff(f_xy_xp,x);
f_xy_yp = diff(f_xy,y);
f_xy_ypp = diff(f_xy_yp,y);
%%
hF = figure;
hA = axes(hF,'NextPlot','add');
hA.XLabel.String = 'X';
hA.YLabel.String = 'Y';
fsurf(f_xy,[-10 10 -10 10])
% Yes

%% Question 10


%% Questions 15-20
% Q15
% max( -0.25*mu.'*A*(Q^-1)*A.'*mu - b.'*mu )

% Q16
syms x1 x2
Q = [3 1; 1 3];
eig(Q)
f = @(x1,x2) [x1 x2]*Q*[x1 x2].';
f_sym = sym(f);
hessian(f_sym)
%%
% A*x <= b
A = [5 1; -4 4];
b = [-5 3].';

%%
% Q17
x0 = [0 0].';
f = @(x) x.'*Q*x;
x_opt = fmincon(f,x0,A,b);
disp(x_opt)

% Q18 - optimal solution to the primal
f_opt = f(x_opt);
disp(['primal solution = ' num2str(f_opt)])
%%
hF = figure;
hA = axes('Parent',hF,'NextPlot','add');
hA.XLabel.String = 'x1';
hA.YLabel.String = 'x2';
hA.ZLabel.String = 'f';
f = @(x1,x2) 3*x1.^2 + 2*x1.*x2 + 3*x2.^2;
fimplicit(f,[-4 4])

%% 
hF = figure;
hA = axes('Parent',hF,'NextPlot','add');
f = @(x1,x2) 3*x1.^2 + 2*x1.*x2 + 3*x2.^2;
fsurf(f)
scatter3(x_opt(1),x_opt(2),f_opt,'o','red','filled')
% fplot3(@(x1)x1,@(x2)x2,f([x1 x2].'))

%% Q19 - optimal solution to the dual
f_dual = @(mu) -0.25*mu.'*A*(Q^-1)*A.'*mu - b.'*mu;
H = 0.5*A*(Q^-1)*A.';
f = b;
lb = [0 0];
ub = [Inf Inf];

mu_opt = quadprog(H,f,[],[],[],[],lb,ub);
f_dual_opt = f_dual(mu_opt);
disp(['dual solution = ' num2str(f_dual_opt)]),

%% Q20 - absolute value of duality gap
disp(['duality gap = ' num2str(round(abs(f_opt - f_dual_opt)))]);

%% Questions 21-22
% Q21
syms q11 q22 q12 q21 x1 x2
Q = [q11 q12; q21 q22];
f = -x.'*Q*x;
hessian(f,[x1 x2])
[~,lambda] = eig(hessian(f,[x1 x2]));

subs(lambda,{'q11','q22','q12','q21'},{-1,-1,0,0})
% no, Q determines convexity.

% Q22


%% Questions 23-25




