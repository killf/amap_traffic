# amap_traffic

人是如何判断拥堵程度的？根据相邻两张图像的差异和时间间隔进行判断的，相同时间间隔下场景差异越大，说明车辆速度越快

给定两张图片和时间，估算车速

注意，采样时间是不同的

参考 https://github.com/Cyclocosmia-ricketti/correlation-layer

输入两张图片和时间 输出速度
输入两张图片 
IMG1 + IMG2 -> S / T -> LABEL

根据车速判断:拥堵 缓行 畅通
[h,w] 