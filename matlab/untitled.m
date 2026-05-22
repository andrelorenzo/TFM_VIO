clear; clc; close all;

p.g = 9.80665;
p.rho = 1.225;

p.m = 150;
p.I = diag([20 80 100]);

p.S = 0.25;
p.b = 1.8;
p.c = 0.35;

p.Tmax = 5000;

p.CL0 = 0.05;
p.CLa = 4.5;
p.CLde = 0.35;

p.CD0 = 0.025;
p.K = 0.08;

p.CYb = -0.8;
p.CYdr = 0.25;

p.Clb = -0.12;
p.Clp = -0.5;
p.Clr = 0.08;
p.Clda = 0.08;
p.Cldr = 0.02;

p.Cm0 = 0.02;
p.Cma = -1.2;
p.Cmq = -8.0;
p.Cmde = -1.0;

p.Cnb = 0.12;
p.Cnp = -0.03;
p.Cnr = -0.25;
p.Cnda = 0.01;
p.Cndr = -0.08;

V0 = 240;
Vz0 = 60;
gamma0 = asin(Vz0/V0);

trim0 = [deg2rad(2); deg2rad(-1); 0.5];
opts = optimset('Display','iter','TolX',1e-8,'TolFun',1e-8);
trim = fminsearch(@(z)trimCost(z,V0,gamma0,p),trim0,opts);

alpha0 = trim(1);
de0 = trim(2);
thr0 = trim(3);

x0 = zeros(12,1);
x0(1) = V0*cos(alpha0);
x0(3) = V0*sin(alpha0);
x0(8) = gamma0 + alpha0;
x0(12) = -1000;

u0 = [de0; 0; 0; thr0];

fprintf('\nTrim encontrado:\n');
fprintf('alpha = %.3f deg\n',rad2deg(alpha0));
fprintf('theta = %.3f deg\n',rad2deg(x0(8)));
fprintf('de = %.3f deg\n',rad2deg(de0));
fprintf('throttle = %.3f\n',thr0);

[A,B] = linmod_fd(@uav6dofDyn,x0,u0,p);

disp('Autovalores del modelo lineal:');
disp(eig(A));

Ts_inner = 1/350;
Ts_outer = 1/20;

fprintf('\nFrecuencia recomendada inicial:\n');
fprintf('Bucle interno: %.1f Hz\n',1/Ts_inner);
fprintf('Bucle externo: %.1f Hz\n',1/Ts_outer);

tspan = [0 10];
[t,x] = ode45(@(t,x)uav6dofDyn(t,x,u0,p),tspan,x0);

figure;
plot(t,x(:,7:9)*180/pi);
grid on;
xlabel('t [s]');
ylabel('Euler [deg]');
legend('\phi','\theta','\psi');

function J = trimCost(z,V,gamma,p)
alpha = z(1);
de = z(2);
thr = min(max(z(3),0),1);
x = zeros(12,1);
x(1) = V*cos(alpha);
x(3) = V*sin(alpha);
x(8) = gamma + alpha;
u = [de; 0; 0; thr];
xdot = uav6dofDyn(0,x,u,p);
J = xdot(1)^2 + xdot(3)^2 + 10*xdot(5)^2;
end

function xdot = uav6dofDyn(~,x,u,p)
ub = x(1);
vb = x(2);
wb = x(3);
pp = x(4);
qq = x(5);
rr = x(6);
phi = x(7);
theta = x(8);
psi = x(9);

de = u(1);
da = u(2);
dr = u(3);
thr = min(max(u(4),0),1);

V = max(sqrt(ub^2 + vb^2 + wb^2),1e-3);
alpha = atan2(wb,ub);
beta = asin(max(min(vb/V,1),-1));
qbar = 0.5*p.rho*V^2;

CL = p.CL0 + p.CLa*alpha + p.CLde*de;
CD = p.CD0 + p.K*CL^2;
CY = p.CYb*beta + p.CYdr*dr;

Cl = p.Clb*beta + p.Clp*(p.b/(2*V))*pp + p.Clr*(p.b/(2*V))*rr + p.Clda*da + p.Cldr*dr;
Cm = p.Cm0 + p.Cma*alpha + p.Cmq*(p.c/(2*V))*qq + p.Cmde*de;
Cn = p.Cnb*beta + p.Cnp*(p.b/(2*V))*pp + p.Cnr*(p.b/(2*V))*rr + p.Cnda*da + p.Cndr*dr;

L = qbar*p.S*CL;
D = qbar*p.S*CD;
Y = qbar*p.S*CY;

Fx_a = -D*cos(alpha) + L*sin(alpha);
Fy_a = Y;
Fz_a = -D*sin(alpha) - L*cos(alpha);

Fx = Fx_a + thr*p.Tmax;
Fy = Fy_a;
Fz = Fz_a;

Mx = qbar*p.S*p.b*Cl;
My = qbar*p.S*p.c*Cm;
Mz = qbar*p.S*p.b*Cn;

udot = rr*vb - qq*wb + Fx/p.m - p.g*sin(theta);
vdot = pp*wb - rr*ub + Fy/p.m + p.g*sin(phi)*cos(theta);
wdot = qq*ub - pp*vb + Fz/p.m + p.g*cos(phi)*cos(theta);

omega = [pp; qq; rr];
M = [Mx; My; Mz];
omegadot = p.I\(M - cross(omega,p.I*omega));

E = [1 sin(phi)*tan(theta) cos(phi)*tan(theta); 0 cos(phi) -sin(phi); 0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
eulerdot = E*omega;

R = [cos(theta)*cos(psi) sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi) cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi); cos(theta)*sin(psi) sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi) cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi); -sin(theta) sin(phi)*cos(theta) cos(phi)*cos(theta)];
posdot = R*[ub; vb; wb];

xdot = [udot; vdot; wdot; omegadot; eulerdot; posdot];
end

function [A,B] = linmod_fd(f,x0,u0,p)
nx = numel(x0);
nu = numel(u0);
A = zeros(nx,nx);
B = zeros(nx,nu);
epsx = 1e-5;
epsu = 1e-5;
for i = 1:nx
dx = zeros(nx,1);
dx(i) = epsx;
A(:,i) = (f(0,x0+dx,u0,p) - f(0,x0-dx,u0,p))/(2*epsx);
end
for i = 1:nu
du = zeros(nu,1);
du(i) = epsu;
B(:,i) = (f(0,x0,u0+du,p) - f(0,x0,u0-du,p))/(2*epsu);
end
end