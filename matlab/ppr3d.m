clear; close all; clc;

dt = 0.03; % Paso de simulación
Tf = 100;   % Tiempo total máximo de simulación en segundos.
N = round(Tf/dt);   % Número total de iteraciones de la simulación.

lookahead = 0.75;       % Distancia adelantada
searchWindow = 50;     % Número de puntos de la trayectoria que se analizan hacia delante para buscar el punto más cercano.
pathResolution = 0.35;  % Separación aproximada entre puntos consecutivos de la trayectoria interpolada.

ctrl.kp = [1.20; 1.20; 1.00];
ctrl.ki = [0.00; 0.00; 0.00];
ctrl.kd = [0.25; 0.25; 0.20];
ctrl.vMaxXY = 0.75;
ctrl.vMaxZ = 0.45;

tauUAV = 0.25;
animationPause = 0.005;

waypoints = [
    0.0   0.0   0.8
    1.5   0.5   1.6
    3.0   1.8   2.4
    2.0   3.4   1.3
    0.0   4.0   2.8
   -2.0   3.0   1.6
   -3.2   1.0   3.0
   -1.2  -0.8   2.1
    1.0  -2.5   1.2
    3.0  -1.0   2.4
    1.5   0.4   1.4
    0.0   0.0   0.4
];

path = pathFromWaypoints3D(waypoints, pathResolution);
sPath = computeArcLength(path);

p = waypoints(1,:).' + [-0.6; 0.5; 0.25];
v = zeros(3,1);
yaw = 0;

intErr = zeros(3,1);
prevErr = zeros(3,1);
lastIdx = 1;

pHist = zeros(N,3);
pRefHist = zeros(N,3);

fig = figure;
set(fig, 'Color', 'w');
hold on; grid on; axis equal;

plot3(path(:,1), path(:,2), path(:,3), '--', 'LineWidth', 1.2);
plot3(waypoints(:,1), waypoints(:,2), waypoints(:,3), 'ko-', 'LineWidth', 1.3, 'MarkerSize', 6, 'MarkerFaceColor', 'y');

hTrail = animatedline('LineWidth', 2.0);
hRef = plot3(NaN, NaN, NaN, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hDrone = createDroneGraphic(p, yaw);
hLine = plot3([p(1) p(1)], [p(2) p(2)], [p(3) p(3)], '-.', 'LineWidth', 1.0);

xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
title('Pure Pursuit 3D basado en waypoints');
legend('Trayectoria interpolada', 'Waypoints', 'Trayectoria UAV', 'Lookahead', 'Location', 'best');

xlim([min(waypoints(:,1))-1 max(waypoints(:,1))+1]);
ylim([min(waypoints(:,2))-1 max(waypoints(:,2))+1]);
zlim([0 max(waypoints(:,3))+1]);
view(45, 25);

for k = 1:N

    [pRef, lastIdx, finished] = purePursuitTarget3D(p, path, sPath, lookahead, lastIdx, searchWindow);

    [vCmd, intErr, prevErr] = pidPosition3D(pRef, p, intErr, prevErr, ctrl, dt);

    v = v + (vCmd - v)*dt/tauUAV;
    p = p + v*dt;
    p(3) = max(p(3), 0);

    if norm(v(1:2)) > 1e-3
        yaw = atan2(v(2), v(1));
    end

    pHist(k,:) = p.';
    pRefHist(k,:) = pRef.';

    if mod(k, 3) == 0
        addpoints(hTrail, p(1), p(2), p(3));
        set(hRef, 'XData', pRef(1), 'YData', pRef(2), 'ZData', pRef(3));
        set(hLine, 'XData', [p(1) pRef(1)], 'YData', [p(2) pRef(2)], 'ZData', [p(3) pRef(3)]);
        updateDroneGraphic(hDrone, p, yaw);
        title(sprintf('Pure Pursuit 3D con waypoints | t = %.2f s | error = %.2f m', k*dt, norm(pRef - p)));
        drawnow;
        pause(animationPause);
    end

    if finished && norm(p - path(end,:).') < 0.12
        pHist = pHist(1:k,:);
        pRefHist = pRefHist(1:k,:);
        break;
    end
end

figure;
plot3(path(:,1), path(:,2), path(:,3), '--', 'LineWidth', 1.2);
hold on; grid on; axis equal;
plot3(waypoints(:,1), waypoints(:,2), waypoints(:,3), 'ko-', 'LineWidth', 1.3, 'MarkerSize', 6, 'MarkerFaceColor', 'y');
plot3(pHist(:,1), pHist(:,2), pHist(:,3), 'LineWidth', 2.0);
plot3(waypoints(1,1), waypoints(1,2), waypoints(1,3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(waypoints(end,1), waypoints(end,2), waypoints(end,3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
legend('Trayectoria interpolada', 'Waypoints', 'Trayectoria UAV', 'Inicio', 'Final', 'Location', 'best');
title('Seguimiento de waypoints mediante Pure Pursuit 3D');

function path = pathFromWaypoints3D(waypoints, resolution)

path = [];

for i = 1:size(waypoints,1)-1
    p0 = waypoints(i,:);
    p1 = waypoints(i+1,:);
    L = norm(p1 - p0);
    n = max(2, ceil(L/resolution));
    alpha = linspace(0, 1, n).';
    segment = (1 - alpha).*p0 + alpha.*p1;

    if i > 1
        segment = segment(2:end,:);
    end

    path = [path; segment];
end

end

function sPath = computeArcLength(path)

ds = sqrt(sum(diff(path,1,1).^2, 2));
sPath = [0; cumsum(ds)];

end

function [pRef, idx, finished] = purePursuitTarget3D(p, path, sPath, lookahead, lastIdx, searchWindow)

idx0 = max(lastIdx - 10, 1);
idx1 = min(lastIdx + searchWindow, size(path,1));

d = sqrt(sum((path(idx0:idx1,:) - p.').^2, 2));
[~, localIdx] = min(d);

idx = idx0 + localIdx - 1;

sTarget = sPath(idx) + lookahead;

if sTarget >= sPath(end)
    pRef = path(end,:).';
    finished = true;
    return;
end

idxTarget = find(sPath >= sTarget, 1, 'first');

s0 = sPath(idxTarget - 1);
s1 = sPath(idxTarget);

alpha = (sTarget - s0)/(s1 - s0 + eps);

p0 = path(idxTarget - 1,:).';
p1 = path(idxTarget,:).';

pRef = (1 - alpha)*p0 + alpha*p1;

finished = false;

end

function [vCmd, intErr, prevErr] = pidPosition3D(pRef, p, intErr, prevErr, ctrl, dt)

err = pRef - p;

intErr = intErr + err*dt;

derErr = (err - prevErr)/dt;

vCmd = ctrl.kp.*err + ctrl.ki.*intErr + ctrl.kd.*derErr;

vXY = norm(vCmd(1:2));

if vXY > ctrl.vMaxXY
    vCmd(1:2) = ctrl.vMaxXY*vCmd(1:2)/vXY;
end

vCmd(3) = min(max(vCmd(3), -ctrl.vMaxZ), ctrl.vMaxZ);

prevErr = err;

end

function hDrone = createDroneGraphic(p, yaw)

L = 0.28;

R = [cos(yaw) -sin(yaw) 0; sin(yaw) cos(yaw) 0; 0 0 1];

armX = [p - R*[L;0;0], p + R*[L;0;0]];
armY = [p - R*[0;L;0], p + R*[0;L;0]];

hDrone.armX = plot3(armX(1,:), armX(2,:), armX(3,:), 'LineWidth', 3);
hDrone.armY = plot3(armY(1,:), armY(2,:), armY(3,:), 'LineWidth', 3);
hDrone.body = plot3(p(1), p(2), p(3), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

end

function updateDroneGraphic(hDrone, p, yaw)

L = 0.28;

R = [cos(yaw) -sin(yaw) 0; sin(yaw) cos(yaw) 0; 0 0 1];

armX = [p - R*[L;0;0], p + R*[L;0;0]];
armY = [p - R*[0;L;0], p + R*[0;L;0]];

set(hDrone.armX, 'XData', armX(1,:), 'YData', armX(2,:), 'ZData', armX(3,:));
set(hDrone.armY, 'XData', armY(1,:), 'YData', armY(2,:), 'ZData', armY(3,:));
set(hDrone.body, 'XData', p(1), 'YData', p(2), 'ZData', p(3));

end