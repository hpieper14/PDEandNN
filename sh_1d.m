% This solves the 1D Swift Hohenberg Equation using a chebyshev
% approximation and saves the solution data to a .mat file

clear all 

% fix domain and discretization parameters
xmax=20;
dom = [0 xmax];
tspan = 0:.025:5;
N=511;
step=xmax/N;
spatial_vec=0:step:xmax;

N=max(size(spatial_vec));


% Initial data

%u0=chebfun(@(x)  exp(-(x-10).^2), dom);
u0=chebfun(@(x)  - exp(-5*(x-5).^2)-exp(-5*(x-15).^2), dom);
%u0=chebfun(@(x)  10*sech(x)*exp(-(x-1).^2) , dom);

% construct the differential operator
S1 = spinop(dom, tspan);
S1.lin = @(u) -2*diff(u,2) - diff(u,4);
r = 1; g = 1.2;
S1.nonlin = @(u) (-1 + r)*u + g*u.^2 - u.^3;
S1.init=u0;

% solve the PDE
L = spin(S1, N, 1e-6, 'plot','off');
disp('Have solved the PDE. Now formatting data.')


% plot the PDE
%waterfall(L), xlabel x, ylabel t
%plot(u), view(0,90), axis equal, axis off

% format solution data 
tsteps=max(size(tspan));
data=zeros(N,tsteps);
for k = 1: tsteps
    snapshot=L{k};
    disp(k)
    for i =1:N
            data(i,k) = snapshot(spatial_vec(i));
    end
end

% save the data to .mat
%save('exp1d_HD_even', 'tspan', 'spatial_vec','data')

