% ==========================================================================
clear;
clc;

% 冷启动表
load data               % 
disp('Loading: data.mat');
lines = {
    'd-', ...
    'x-',  ...
    '<-', 's-', '*-', '^-'};
colors = {
    [0 0 0.5], ...
    [1 0.2 0],  ...
    [0 0 1],    [0 1 1],    [1 0 0],    [0 0.5 1]};
sub = [
    511, 512, 513, 514, 515];

a = 'f55';       % 'taobao'拼写正确则为运行t3数据库，否则运行a1数据库
if strcmp(a, 'f55')
    dataset = f55_test;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [0, 27]
    [0, 5.3]};
else
    dataset = g55_test;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [0, 27]
    [0, 5.6]};    
end
    
data = dataset{3};
x = 1:1:34;
for j = [0, 5]      % 前5条线一个图，后5条线一个图。
    figure();
    set(gca,'FontSize',15);
    for i = [1, 2, 3, 4, 5]
        subplot(sub(i));
        plot(x, data{i+j}, 'LineWidth', 2);   % 'Color',colors{i}, 
        xlabel('Interval')
        ylabel('Probability')
    end
end

% 一次性绘制出foursquare上10个聚类的曲线后，
% 发现cluster-3/4/9的区分性比较强。且最大均是0.2左右。


