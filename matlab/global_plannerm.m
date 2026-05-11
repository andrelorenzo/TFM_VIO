clear; close all; clc;

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

path = [];
step = 0.2;

for i = 1:1:size(waypoints,1)-2

    p0 = waypoints(i,:);
    p1 = waypoints(i+1,:);
    p2 = waypoints(i+2,:);

    for t = 0:step:1
        point = (1-t)^2*p0 + 2*(1-t)*t*p1 + t^2*p2;
        path = [path; point];
    end

end

figure;
hold on; grid on; axis equal;

plot3(waypoints(:,1), waypoints(:,2), waypoints(:,3), 'ko-', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', 'y');
plot3(path(:,1), path(:,2), path(:,3), 'b-', 'LineWidth', 1.5);
scatter3(path(:,1), path(:,2), path(:,3), 18, 'filled');

xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
title('Puntos generados a partir de los waypoints');
legend('Waypoints originales', 'Trayectoria generada', 'Puntos generados', 'Location', 'best');
view(45, 25);

for i = 1:size(waypoints,1)
    text(waypoints(i,1), waypoints(i,2), waypoints(i,3), sprintf('  W%d', i), 'FontSize', 10, 'FontWeight', 'bold');
end