clear; close all; clc;

pc = pcread("pointclouds\aligned_global_cloud.ply");
pc2 = pcdenoise(pc,"NumNeighbors",1000 ...
    );

figure(1);
pcshow(pc);
figure(2);
pcshow(pc2);