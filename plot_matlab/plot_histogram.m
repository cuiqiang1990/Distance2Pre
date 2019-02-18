% ==========================================================================
clear

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
    121, 122];

a = 'f5';       % 'taobao'拼写正确则为运行t3数据库，否则运行a1数据库
if strcmp(a, 'f55')
    dataset = f55_test;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [5, 27]
    [1, 5.3]};
else
    dataset = g55_test;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [5, 28]
    [1, 5.5]};    
end

figure();
set(gca,'FontSize',15);
x = [1, 2, 3, 4];
set(gca, 'XTick', x);   % 指定x轴刻度标识
    
for num = [1, 2]
    name = data_evaluation{num};    % 评价指标
    data = dataset{num};

    subplot(sub(num));
    bar(data, 0.8)
    % bar(data,style, color, width)
    
    xlabel('top-\itk');     % k用斜体表示。
    labels = data_at_nums;  % 这个得用{'a', 'b'}，不行['a', 'b']
    set(gca, 'XTickLabel', labels);   % 指定x轴显示标识  
    xlim([0.4 4.6])
    ylabel(name)
    ylim(ylims{num})
end

%hl = legend(data_method);       % 各种方法名
%set(hl, 'Location', 'NorthOutside', 'Orientation', 'horizontal','Box', 'off');





