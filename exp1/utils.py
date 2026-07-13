import os
# (语法) import 表示“导入模块”。
# os 是 Python 标准库模块，提供和操作系统交互的功能。

# 后面用 os.popen(...) 执行终端命令，读取终端宽度。

import sys
# sys 是 Python 标准库模块，提供和 Python 解释器/标准输入输出相关的功能。

# 后面用 sys.stdout.write(...) 手动往终端输出字符。

import time
# time 是 Python 标准库模块，提供时间相关功能。

# 后面用 time.time() 记录进度条每一步和总共用了多久。

##############
# 获取终端宽度 #
##############        
_, term_width = os.popen('stty size', 'r').read().split()
# os.popen('stty size', 'r') 会在终端里执行 stty size 命令，并读取它的输出。
# - stty size 通常会输出两项：终端高度 (多少行), 终端宽度 (多少列)
#   e.g. 可能输出 "24 120" 表示终端高 24 行，宽 120 列。
# - 可以拆开看：
#   - os.popen('stty size', 'r').read() 读取命令输出的字符串
#     得到一个字符串，例如 "24 120\n"
#   - "24 120\n".split() 按空白字符拆成列表
#     得到一个列表，例如 ['24', '120']
#   - _, term_width = ['24', '120']
#     把列表里的两个元素分别赋值给 _ 和 term_width。
#
# _, term_width = ... 表示把列表里的两个值分别赋给两个变量：
# - _ 接住第一个值，也就是终端高度。这里后面不用它，所以用 _ 表示“我不关心这个值”
# - term_width 接住第二个值，也就是终端宽度，e.g. 字符串 '120'。

term_width = int(term_width)
# int(term_width) 把字符串 '120' 转换成整数 120，后面才能用它做减法计算。

#####################
# 进度条相关的全局变量 #
#####################
TOTAL_BAR_LENGTH = 65.
# TOTAL_BAR_LENGTH 是进度条本体的长度。
# 65. 是浮点数 float，等价于 65.0。

last_time = time.time()
# last_time 记录上一次刷新进度条的时间。
# time.time() 返回当前时间戳，单位是秒。

begin_time = last_time
# begin_time 记录当前进度条开始的时间。
# 一开始还没进入 progress_bar，所以先让 begin_time 和 last_time 相同。

def progress_bar(current, total, msg=None):
    # 参数含义：
    # - current：当前是第几个 batch，从 0 开始
    # - total：总共有多少个 batch
    # - msg：额外显示的信息，例如 loss、accuracy；默认是 None
    #
    # 在 train.py 里，它被这样调用：
    # progress_bar(idx, len(trainloader), 'Loss: ... | Acc: ...')

    global last_time, begin_time
    # (语法) global 表示使用函数外面的全局变量。
    # 如果没有 global，函数里给 last_time、begin_time 赋值时，
    # Python 会以为它们是函数内部新建的局部变量。
    #
    # 这里需要修改外面的 last_time 和 begin_time，
    # 所以要写 global last_time, begin_time。

    if current == 0:
        begin_time = time.time() 
        # 如果 current == 0，说明这是一个 epoch 里的第一个 batch。
        # 这时重新记录 begin_time，表示新进度条从这里开始计时。

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    # cur_len 表示进度条里已经完成的长度。
    # - current / total 接近 0，说明刚开始，cur_len 很小；
    # - current / total 接近 1，说明快结束，cur_len 接近 TOTAL_BAR_LENGTH。

    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    # rest_len 表示进度条里还没完成的长度。
    # 减 1 是为了给中间的 '>' 箭头留一个位置。

    ############################ 打印进度条本体 ################################
    sys.stdout.write(' [')
    # sys.stdout.write(...) 是往终端输出字符串。
    # 和 print(...) 不同，write 默认不会自动换行。

    for i in range(cur_len): # 循环 cur_len 次，i 是循环变量。
        sys.stdout.write('=') # 已完成的部分用 '=' 表示。

    sys.stdout.write('>') # '>' 表示当前进度位置。

    for i in range(rest_len):
        sys.stdout.write('.') # 未完成的部分用 '.' 表示。

    sys.stdout.write(']') # 打印进度条右边界。

    ############################ 计算耗时 ######################################
    cur_time = time.time() # 当前时间。

    step_time = cur_time - last_time
    # step_time 是距离上一次刷新进度条过去了多久。
    # 可以理解为“当前 batch 附近这一步用了多久”。

    last_time = cur_time
    # 更新 last_time，供下一次 progress_bar 调用时计算 step_time。

    tot_time = cur_time - begin_time
    # tot_time 是从当前进度条开始到现在，总共过去了多久。
    # 也就是“从 epoch 开始到现在总共用了多久”。

    ############################ 拼接进度条后面的文字 ###########################
    L = []
    # L 是一个列表，用来暂存要显示的几段文字。

    L.append('  Step: %s' % format_time(step_time))
    # append(...) 是列表对象的方法，表示往列表末尾添加一个元素。
    # %s 是字符串格式化占位符，会被 format_time(step_time) 的返回值替换。

    L.append(' | Tot: %s' % format_time(tot_time))
    # 添加总耗时文字。

    if msg: # 如果调用 progress_bar(...) 时传入了 msg，就把它也加到显示内容里。
        L.append(' | ' + msg)
        # 例如 msg 可能是 "Loss: 1.234 | Acc: 56.789% (123/456)"

    msg = ''.join(L)
    # ''.join(L) 表示把列表 L 里的字符串连接成一个完整字符串。
    # 这里用空字符串 '' 连接，表示中间不额外加分隔符。

    sys.stdout.write(msg) # 把 Step、Tot、Loss、Acc 等文字写到终端。

    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')
        # 根据终端宽度补空格。
        # 目的：覆盖上一轮进度条可能残留的字符，让显示更整齐。

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
        # \b 是退格字符，表示光标往左退一格。
        # 这里通过连续输出很多个 \b，把光标移动回进度条中间附近。

    sys.stdout.write(' %d/%d ' % (current + 1, total))
    # 在进度条中间写当前进度数字。
    # current 从 0 开始，所以显示时用 current + 1。
    # 例如 3/391 表示总共 391 个 batch，目前处理到第 3 个。

    if current < total - 1:
        sys.stdout.write('\r')
        # \r 是回车符，表示把光标移动到当前行开头，但不换行。
        # 这样下一次刷新进度条时，可以覆盖同一行内容。

    else:
        sys.stdout.write('\n')
        # 如果已经是最后一个 batch，就输出换行，结束这一条进度条。

    sys.stdout.flush()
    # flush() 强制把缓冲区里的内容立刻显示到终端。
    # 如果不 flush，有些环境可能不会马上看到进度条更新。


def format_time(seconds):
    # format_time(...) 是一个函数，用来把秒数转换成更易读的时间字符串。
    #
    # 参数含义：
    # - seconds：一个秒数，可以是整数或浮点数
    #
    # 返回值：
    # - 一个字符串，例如 '3s'、'42ms'、'1m2s'

    ############################ 把秒数拆成天/小时/分钟/秒/毫秒 ##################
    days = int(seconds / 3600 / 24)
    # 计算有多少整天，1 天 = 24 小时 = 24 * 3600 秒。
    seconds = seconds - days * 3600 * 24
    # 减去已经算进 days 的秒数，剩下的继续拆。

    hours = int(seconds / 3600)
    # 计算有多少整小时， 1 小时 = 3600 秒。
    seconds = seconds - hours * 3600
    # 减去已经算进 hours 的秒数。

    minutes = int(seconds / 60)
    # 计算有多少整分钟。
    seconds = seconds - minutes * 60
    # 减去已经算进 minutes 的秒数，1 分钟 = 60 秒。

    secondsf = int(seconds)
    # secondsf 是整数秒。
    # 变量名里的 f 可以理解成 fixed/integer seconds，这里表示秒的整数部分。
    seconds = seconds - secondsf
    # 剩下的小数部分用于计算毫秒。

    millis = int(seconds * 1000)
    # 把剩下的小数秒转换成毫秒。

    ############################ 拼接时间字符串 ################################
    f = '' # f 是最终要返回的时间字符串。

    i = 1 # i 用来控制最多显示两个时间单位。
    # 例如 1h2m、3s45ms，而不是把天/小时/分钟/秒/毫秒全部显示出来。

    if days > 0:
        f += str(days) + 'D'
        # 如果有天数，就拼上类似 2D。
        # str(days) 把整数转换成字符串。

        i += 1
        # 已经显示了一个单位，计数加 1。

    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
        # 如果有小时，并且目前显示的单位还不超过两个，就拼上小时。

    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
        # 如果有分钟，并且还可以显示，就拼上分钟。

    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
        # 如果有整数秒，并且还可以显示，就拼上秒。

    if millis > 0 and i <= 2:
        f += str(millis).zfill(3) + 'ms'
        i += 1
        # 如果有毫秒，并且还可以显示，就拼上毫秒。
        # zfill(3) 表示左边补 0 到 3 位，例如 7 -> '007'。

    if f == '':
        f = '0ms'
        # 如果时间太短，前面什么都没拼上，就返回 0ms。

    return f # 返回格式化后的时间字符串。
