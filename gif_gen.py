# 读取imgs文件夹内的png文件，生成gif动画
# 生成的gif动画保存在gif文件夹内
# 生成的gif动画名称为output.gif
# 生成的gif动画帧率为10
# 生成的gif动画循环播放
# 生成的gif动画尺寸为400x400
# 生成的gif动画背景色为白色
# 生成的gif动画前景色为黑色
# 生成的gif动画字体为arial.ttf
# 生成的gif动画字体大小为20
# 生成的gif动画字体颜色为黑色

import imageio.v2 as imageio
import os

# 读取imgs文件夹内的png文件
png_files = os.listdir('imgs')
png_files = [f for f in png_files if f.endswith('.png')]
# png_files.sort()
# 图片名称排序 文件名类似于plot_2024-11-14 23-30-30_02.png，根据最后一个_后的数字排序
png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 生成gif动画保存在gif文件夹内
if not os.path.exists('gif'):
    os.makedirs('gif')

# 生成的gif动画名称为output.gif
output_file = 'gif/output.gif'

# 生成的gif动画帧率为10
fps = 10

# 生成的gif动画循环播放
loop = 0

# 生成的gif动画尺寸为400x400
size = (640, 480)

# 生成的gif动画背景色为白色
bg_color = (255, 255, 255)

# 生成的gif动画前景色为黑色
fg_color = (0, 0, 0)

# 生成的gif动画字体为arial.ttf
font = 'arial.ttf'

# 生成的gif动画字体大小为20
font_size = 20

# 生成的gif动画字体颜色为黑色
font_color = (0, 0, 0)

# 生成gif动画

with imageio.get_writer(output_file, mode='I', fps=fps, loop=loop) as writer:
    for filename in png_files:
        image = imageio.imread(os.path.join('imgs', filename))
        writer.append_data(image)
print('Gif animation generated successfully!')
